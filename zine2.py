import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained('./custom_model')
tokenizer = GPT2Tokenizer.from_pretrained('./custom_model')

# Ensure pad_token is set
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to the end of sequence token

# Function to generate text
def generate_text(prompt):
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=128)
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'], 
            attention_mask=inputs['attention_mask'],  # Explicitly pass attention mask
            max_length=150,  # Adjust this as needed
            num_return_sequences=1
        )
    
    # Decode the generated text
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated

# Main execution
if __name__ == "__main__":
    prompt = input("Enter your prompt: ")
    result = generate_text(prompt)
    print("Generated Completion:", result)
