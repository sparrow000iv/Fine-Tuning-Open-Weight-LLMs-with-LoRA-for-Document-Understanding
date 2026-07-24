"""
Simple inference script for running the fine-tuned model on new documents.

Usage:
    python scripts/inference.py --model checkpoints/mistral_lora_r16/final_model --input data/sample_doc.txt
    
Or for batch inference:
    python scripts/inference.py --model checkpoints/mistral_lora_r16/final_model --input-dir data/test_docs/
"""

import os
import json
import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_model(model_path, use_qlora=False):
    """Load the fine-tuned model and tokenizer."""
    logger.info(f"Loading model from {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    
    if use_qlora:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["quantization_config"] = bnb_config
    
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()
    
    return model, tokenizer


def extract_fields(model, tokenizer, document_text, instruction=None):
    """
    Extract structured fields from a document.
    
    Args:
        model: The fine-tuned model
        tokenizer: Tokenizer
        document_text: OCR text of the document
        instruction: Optional custom instruction (defaults to generic extraction)
    
    Returns:
        dict: Extracted fields as JSON
    """
    if instruction is None:
        instruction = "Extract all fields from this document and return as JSON."
    
    prompt = f"<s>[INST] {instruction}\n\n{document_text} [/INST]"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # Try to parse as JSON
    try:
        result = json.loads(generated.strip())
        return result
    except json.JSONDecodeError:
        # Try to extract JSON from the text
        start = generated.find('{')
        end = generated.rfind('}')
        if start != -1 and end != -1:
            try:
                return json.loads(generated[start:end+1])
            except:
                pass
        
        logger.warning(f"Could not parse JSON from model output: {generated[:200]}")
        return {"raw_output": generated, "parse_error": True}


def main():
    parser = argparse.ArgumentParser(description='Run inference on documents')
    parser.add_argument('--model', type=str, required=True, help='Path to model')
    parser.add_argument('--input', type=str, help='Single input file')
    parser.add_argument('--input-dir', type=str, help='Directory of input files')
    parser.add_argument('--output', type=str, default='output.json', help='Output JSON file')
    parser.add_argument('--qlora', action='store_true', help='Use QLoRA loading')
    args = parser.parse_args()
    
    model, tokenizer = load_model(args.model, args.qlora)
    
    results = []
    
    if args.input:
        # Single file
        with open(args.input, 'r') as f:
            doc_text = f.read()
        
        logger.info(f"Processing {args.input}")
        extracted = extract_fields(model, tokenizer, doc_text)
        results.append({
            'file': args.input,
            'extracted': extracted
        })
        
    elif args.input_dir:
        # Batch processing
        input_dir = Path(args.input_dir)
        for file_path in sorted(input_dir.glob('*.txt')):
            with open(file_path, 'r') as f:
                doc_text = f.read()
            
            logger.info(f"Processing {file_path.name}")
            extracted = extract_fields(model, tokenizer, doc_text)
            results.append({
                'file': file_path.name,
                'extracted': extracted
            })
    else:
        logger.error("Must specify either --input or --input-dir")
        return
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {args.output}")
    
    # Print summary
    for r in results:
        print(f"\n{r['file']}:")
        print(json.dumps(r['extracted'], indent=2))


if __name__ == '__main__':
    main()
