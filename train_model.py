import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

# Step 2: Load the dataset (assuming it's named 'papers.csv')
dataset_path = 'papers.csv'  # This should match the name of the file uploaded
data = pd.read_csv(dataset_path)

# Step 3: Prepare the dataset for training
with open("training_data.txt", "w") as f:
    for title in data['title']:  # Adjust if your dataset structure is different
        f.write(f"{title}\n")

# Step 4: Initialize the GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')  # Use pretrained GPT-2 model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', use_fast=True)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

# Step 5: Prepare the dataset and Data Collator
train_dataset = TextDataset(tokenizer=tokenizer, file_path="training_data.txt", block_size=128)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir="./custom_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
)

# Trainer to handle training
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# Train the model
trainer.train()

# Save the trained model
model.save_pretrained("./custom_model")
tokenizer.save_pretrained("./custom_model")
