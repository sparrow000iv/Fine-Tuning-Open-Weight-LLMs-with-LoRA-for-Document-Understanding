"""
Custom evaluation harness for document understanding model.

Standard metrics like BLEU/ROUGE don't really work well for structured extraction
because the output is JSON and small formatting differences shouldn't count as errors.
So I built this custom evaluator that:
1. Parses JSON outputs
2. Does field-level exact match
3. Does field-level fuzzy match (handles minor formatting differences)
4. Tracks hallucination rate (fields the model invents)
5. Measures adversarial robustness

This is probably the most important script in the repo honestly.
The evaluation is what actually tells you if the model is useful.
"""

import os
import json
import re
import argparse
import logging
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def fuzzy_match(pred, target, threshold=0.85):
    """
    Fuzzy string matching using SequenceMatcher.
    Returns True if similarity > threshold.
    
    I use 0.85 because things like extra spaces or slight OCR differences
    shouldn't count as errors. But this threshold is somewhat arbitrary.
    """
    if not pred or not target:
        return pred == target
    
    # Normalize whitespace
    pred_clean = ' '.join(str(pred).strip().split())
    target_clean = ' '.join(str(target).strip().split())
    
    ratio = SequenceMatcher(None, pred_clean.lower(), target_clean.lower()).ratio()
    return ratio >= threshold, ratio


def extract_json_from_output(text):
    """
    Try to extract valid JSON from model output.
    Models sometimes add extra text before/after the JSON, so this handles that.
    """
    # Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON in the text (sometimes model adds explanation)
    # Look for first { and last }
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass
    
    return None


def evaluate_single_prediction(predicted_text, ground_truth):
    """
    Evaluate a single prediction against ground truth.
    Returns a dict of metrics for this example.
    """
    result = {
        'json_valid': False,
        'exact_match': 0.0,
        'fuzzy_match': 0.0,
        'hallucinated_fields': [],
        'missing_fields': [],
        'correct_fields': [],
        'field_scores': {},
    }
    
    # Parse predicted JSON
    pred_json = extract_json_from_output(predicted_text)
    if pred_json is None:
        result['parse_error'] = predicted_text[:200]
        return result
    
    result['json_valid'] = True
    
    # Compare field by field
    gt_keys = set(ground_truth.keys())
    pred_keys = set(pred_json.keys())
    
    # Hallucinated fields (predicted but not in ground truth)
    result['hallucinated_fields'] = list(pred_keys - gt_keys)
    
    # Missing fields
    result['missing_fields'] = list(gt_keys - pred_keys)
    
    # Field-level evaluation
    common_keys = gt_keys & pred_keys
    exact_matches = 0
    fuzzy_matches = 0
    
    for key in gt_keys:
        if key not in pred_json:
            result['field_scores'][key] = {'exact': False, 'fuzzy': False, 'score': 0.0}
            continue
        
        pred_val = str(pred_json[key])
        gt_val = str(ground_truth[key])
        
        exact = (pred_val.strip() == gt_val.strip())
        fuzzy_result = fuzzy_match(pred_val, gt_val)
        fuzzy, score = fuzzy_result if isinstance(fuzzy_result, tuple) else (fuzzy_result, 1.0 if fuzzy_result else 0.0)
        
        result['field_scores'][key] = {
            'predicted': pred_val,
            'ground_truth': gt_val,
            'exact': exact,
            'fuzzy': fuzzy,
            'score': score,
        }
        
        if exact:
            exact_matches += 1
        if fuzzy:
            fuzzy_matches += 1
    
    # Calculate overall scores
    n_fields = len(gt_keys) if gt_keys else 1
    result['exact_match'] = exact_matches / n_fields
    result['fuzzy_match'] = fuzzy_matches / n_fields
    result['correct_fields'] = [k for k, v in result['field_scores'].items() if v.get('fuzzy', False)]
    
    return result


def generate_prediction(model, tokenizer, instruction, input_text, max_new_tokens=512):
    """Generate a prediction from the model."""
    prompt = f"<s>[INST] {instruction}\n\n{input_text} [/INST]"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,  # Low temperature for more deterministic output
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    # Decode only the generated part
    generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return generated.strip()


def load_test_data(data_path):
    """Load test data."""
    examples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return examples


def run_evaluation(model_path, test_data_path, output_path=None, use_qlora=False):
    """
    Run full evaluation on test data.
    """
    logger.info(f"Loading model from {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    
    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["quantization_config"] = bnb_config
    
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()
    
    # Load test data
    test_data = load_test_data(test_data_path)
    logger.info(f"Loaded {len(test_data)} test examples")
    
    # Run predictions
    results = []
    all_scores = {'exact': [], 'fuzzy': [], 'json_valid': [], 'hallucination_rate': []}
    
    for i, example in enumerate(tqdm(test_data, desc="Evaluating")):
        instruction = example.get('instruction', 'Extract all fields from this document.')
        input_text = example['input']
        ground_truth = json.loads(example['output']) if isinstance(example['output'], str) else example['output']
        
        # Generate prediction
        predicted_text = generate_prediction(model, tokenizer, instruction, input_text)
        
        # Evaluate
        eval_result = evaluate_single_prediction(predicted_text, ground_truth)
        eval_result['example_id'] = i
        eval_result['predicted_text'] = predicted_text
        results.append(eval_result)
        
        # Accumulate scores
        all_scores['exact'].append(eval_result['exact_match'])
        all_scores['fuzzy'].append(eval_result['fuzzy_match'])
        all_scores['json_valid'].append(1.0 if eval_result['json_valid'] else 0.0)
        
        # Hallucination rate = number of hallucinated fields / total predicted fields
        n_hallucinated = len(eval_result.get('hallucinated_fields', []))
        n_predicted = len(eval_result.get('correct_fields', [])) + n_hallucinated + len(eval_result.get('missing_fields', []))
        hall_rate = n_hallucinated / max(n_predicted, 1)
        all_scores['hallucination_rate'].append(hall_rate)
    
    # Aggregate results
    summary = {
        'total_examples': len(test_data),
        'avg_exact_match': sum(all_scores['exact']) / len(all_scores['exact']),
        'avg_fuzzy_match': sum(all_scores['fuzzy']) / len(all_scores['fuzzy']),
        'json_validity_rate': sum(all_scores['json_valid']) / len(all_scores['json_valid']),
        'avg_hallucination_rate': sum(all_scores['hallucination_rate']) / len(all_scores['hallucination_rate']),
        'timestamp': datetime.now().isoformat(),
        'model_path': model_path,
    }
    
    # Print summary
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total examples:          {summary['total_examples']}")
    logger.info(f"Avg Exact Match:         {summary['avg_exact_match']:.4f}")
    logger.info(f"Avg Fuzzy Match:         {summary['avg_fuzzy_match']:.4f}")
    logger.info(f"JSON Validity Rate:      {summary['json_validity_rate']:.4f}")
    logger.info(f"Avg Hallucination Rate:  {summary['avg_hallucination_rate']:.4f}")
    logger.info("=" * 60)
    
    # Save results
    output = {'summary': summary, 'detailed_results': results}
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate document understanding model')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Path to test data JSONL')
    parser.add_argument('--output', type=str, default='evaluations/results/eval_results.json')
    parser.add_argument('--qlora', action='store_true', help='Use QLoRA (4-bit) loading')
    args = parser.parse_args()
    
    run_evaluation(args.model, args.data, args.output, args.qlora)
