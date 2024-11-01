import os
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor

# Define paths
output_path = os.path.join(os.curdir(), "model_responses.json")
model_name = 'llava-hf/llava-1.5-7b-hf'

# Load LLaVa
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
processor = AutoProcessor.from_pretrained(model_name)

# Load POPE benchmark dataset
dataset = load_dataset("lmms-lab/POPE", split="default")["test"]
dataset = dataset['test'].filter(lambda x: x['category'] == 'adversarial') # Only use examples generated with adversarial negative sampling


def generate_response(question, image):
    """ Prompt model with question regarding image and generate response.

    Args:
        question (str): question regarding the image content
        image_path (str): PIL image object
    
    Returns:
        response (str): model's response to the question
    """
    inputs = processor(images=image, text=question, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response


responses = []
for idx in range(len(dataset)):
    question = dataset['question'][idx]
    image = dataset['image'][idx]
    response = generate_response(question, image)

    responses.append({
        'question': question,
        'response': response
    })


# Write responses to file
with open(output_path, 'w') as f:
    json.dump(responses, f, indent=4)

print(f"LLaVa's responses have been saved to {output_path}")
