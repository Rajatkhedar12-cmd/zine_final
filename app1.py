import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import streamlit as st

# Load the trained model and tokenizer
@st.cache_resource
def load_model():
    model = GPT2LMHeadModel.from_pretrained('./custom_model')
    tokenizer = GPT2Tokenizer.from_pretrained('./custom_model')
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token to the end of sequence token
    return model, tokenizer

# Function to generate text
def generate_text(prompt, model, tokenizer):
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

# Streamlit App
def main():
    st.title("Text Generation with GPT-2")
    st.write("Enter a prompt below and generate text completions.")

    model, tokenizer = load_model()  # Load the model and tokenizer

    # User input for the prompt
    prompt = st.text_area("Enter your prompt:", height=100)

    if st.button("Generate Text"):
        if prompt:
            with st.spinner("Generating..."):
                result = generate_text(prompt, model, tokenizer)
            st.subheader("Generated Completion:")
            st.write(result)
        else:
            st.error("Please enter a prompt.")

if __name__ == "__main__":
    main()
