import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import streamlit as st

# Load the code examples dataset
def load_code_data(filename='code_examples.csv'):
    try:
        data = pd.read_csv(filename)
        return data
    except FileNotFoundError:
        st.error(f"Error: {filename} not found.")
        return None

# Initialize GPT-2 model and tokenizer for code
@st.cache_resource
def load_model():
    st.write("Loading GPT-2 model for code...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    return model, tokenizer

# Generate code autocompletion using GPT-2
def generate_code_completion(prompt, model, tokenizer, max_length=50):
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to(model.device)

    output = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],  # Explicitly pass attention mask
        max_length=min(max_length + len(inputs['input_ids'][0]), 1024),  # Adjusted max_length
        num_return_sequences=1,
        no_repeat_ngram_size=2,
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# Streamlit App
def main():
    st.title("Code Autocompletion Model")
    st.write("Enter a code prompt below and get autocompletion suggestions.")

    # Load the dataset (optional)
    load_code_data()  # You can remove this if the dataset is not required

    model, tokenizer = load_model()  # Load GPT-2 model and tokenizer

    prompt = st.text_area("Enter your code prompt:", height=100)
    
    if st.button("Generate Completion"):
        if prompt:
            with st.spinner("Generating..."):
                completion = generate_code_completion(prompt, model, tokenizer)
            st.subheader("Generated Code Completion:")
            st.code(completion)
        else:
            st.error("Please enter a prompt.")

if __name__ == "__main__":
    main()
