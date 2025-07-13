---
base_model: unsloth/Qwen2.5-7B-Instruct-bnb-4bit
tags:
- banking
- customer-service
- json-generation
- merged-model
- qwen2.5
- unsloth
- text-generation
license: apache-2.0
language:
- en
pipeline_tag: text-generation
library_name: transformers
---

# QwenInstruct Banking Agent - Merged Model

This is a merged version of the fine-tuned Qwen2.5-7B-Instruct model for banking customer service ticket generation. The LoRA adapters have been merged into the base model for easier deployment.

## Model Details

- **Base Model**: unsloth/Qwen2.5-7B-Instruct-bnb-4bit
- **Fine-tuning Method**: LoRA (merged into base model)
- **Model Format**: 16-bit merged model
- **Model Size**: ~15 GB
- **Training Framework**: Unsloth + TRL

## Training Details

- **LoRA Configuration**:
  - Rank: 32
  - Alpha: 32
  - Dropout: 0.1
  - Target Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

- **Training Configuration**:
  - Dataset Size: 1,000 banking examples
  - Epochs: 5
  - Learning Rate: 1.5e-4
  - Batch Size: 8 (effective)
  - Optimizer: AdamW
  - Weight Decay: 0.01

## Performance Metrics

- **Final Training Loss**: 0.055
- **Final Validation Loss**: 0.072
- **JSON Generation Success Rate**: 100%
- **Test Perplexity**: 8.32

## Usage

### Simple Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the merged model directly
model = AutoModelForCausalLM.from_pretrained(
    "LaythAbuJafar/QwenInstruct_Agent1_Merged",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("LaythAbuJafar/QwenInstruct_Agent1_Merged")
```

### Using with Unsloth

```python
from unsloth import FastLanguageModel

# Load with Unsloth for optimized inference
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="LaythAbuJafar/QwenInstruct_Agent1_Merged",
    max_seq_length=1024,
    dtype=torch.float16,
    load_in_4bit=True,  # Optional: use 4-bit quantization
)

FastLanguageModel.for_inference(model)
```

### Generation Example

```python
# Define the prompt template
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

# Example usage
instruction = "You are a banking customer service assistant. Analyze the customer input and create a complaint/inquiry ticket in valid JSON format. The JSON must include these fields: ticket_type, title, description, severity, department_impacted, service_impacted, supporting_documents, preferred_communication."
user_input = "I received a suspicious email asking for my banking credentials."

prompt = alpaca_prompt.format(
    instruction=instruction,
    input=user_input
)

# Generate response
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.6,
    top_p=0.9,
    repetition_penalty=1.1,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
# Extract JSON from response
json_start = response.find("### Response:") + len("### Response:")
json_response = response[json_start:].strip()
print(json_response)
```

## Expected Output Format

```json
{
    "ticket_type": "complaint|inquiry|assistance",
    "title": "Brief issue description",
    "description": "Detailed customer request explanation",
    "severity": "low|medium|high|critical",
    "department_impacted": "Relevant department",
    "service_impacted": "Affected service",
    "supporting_documents": "Required documents",
    "preferred_communication": "phone|email|chat|not specified"
}
```

## Example Outputs

**Input**: "My credit card payment didn't go through but the money was deducted."
```json
{
    "ticket_type": "complaint",
    "title": "Failed payment with deduction",
    "description": "Customer reports credit card payment failed but money was deducted from account",
    "severity": "high",
    "department_impacted": "Payment Processing",
    "service_impacted": "Credit Card Payments",
    "supporting_documents": "Transaction history, payment confirmation",
    "preferred_communication": "phone"
}
```

## Deployment Tips

1. **Memory Requirements**: ~15 GB for full precision, ~8 GB with 4-bit quantization
2. **Inference Speed**: Use Flash Attention 2 for faster inference
3. **Batch Processing**: Model supports batch inference for multiple tickets
4. **Temperature**: Use 0.5-0.7 for consistent JSON generation

## Limitations

- Specialized for banking domain only
- English language support only
- Requires GPU for optimal performance
- JSON structure is fixed to the trained format

## Citation

```bibtex
@misc{qweninstruct-banking-agent-merged,
  author = {Layth Abu Jafar},
  title = {QwenInstruct Banking Agent - Merged Model},
  year = {2024},
  publisher = {Hugging Face},
  url = {https://huggingface.co/LaythAbuJafar/QwenInstruct_Agent1_Merged}
}
```

## Acknowledgments

- Base model: Qwen Team
- Fine-tuning framework: Unsloth
- Training library: TRL (Transformer Reinforcement Learning)
