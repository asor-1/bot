import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split, KFold
import json
import numpy as np

# Disable tokenizer parallelism to avoid issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load training data from JSON file (absolute path)
train_data_path = "model/intents.json"  # Use forward slash for path
with open(train_data_path, "r") as f:
    t_data = json.load(f)

# Extract texts and intents from training data
texts = [item["text"] for item in t_data]
intents = [item["intent"] for item in t_data]

# Combine text and intent for input sequences
inputs = texts  # Just use texts as inputs for sequence classification

# Convert intents to numerical labels
unique_intents = list(set(intents))
intent_to_id = {intent: idx for idx, intent in enumerate(unique_intents)}
labels = [intent_to_id[intent] for intent in intents]

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# K-fold Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(inputs)))):
    print(f"Training fold {fold + 1}")
    
    train_inputs = [inputs[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_inputs = [inputs[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    
    # Tokenize input sequences
    train_encodings = tokenizer(train_inputs, padding=True, truncation=True, return_tensors="pt")
    val_encodings = tokenizer(val_inputs, padding=True, truncation=True, return_tensors="pt")

    # Convert labels to tensor
    train_labels = torch.tensor(train_labels)
    val_labels = torch.tensor(val_labels)

    # Define custom Dataset class
    class IntentDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
            item['labels'] = self.labels[idx].clone().detach()
            return item

        def __len__(self):
            return len(self.labels)

    # Create instances of Dataset for training and validation
    train_dataset = IntentDataset(train_encodings, train_labels)
    val_dataset = IntentDataset(val_encodings, val_labels)

    # Load pre-trained model
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(unique_intents)).to(device)

    # Update training arguments
    training_args = TrainingArguments(
        output_dir=f"./trained_model_fold_{fold + 1}",
        num_train_epochs=10,  # Reduce to 10 epochs
        per_device_train_batch_size=4,  # Reduce batch size
        per_device_eval_batch_size=4,  # Reduce batch size
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        save_strategy="epoch",  # Save the model at each epoch
        evaluation_strategy="epoch",
        load_best_model_at_end=True,  # Enable loading the best model at the end
        metric_for_best_model="eval_loss",
        use_cpu=not torch.cuda.is_available(),
    )

    # Learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=500, 
        num_training_steps=len(train_dataset) * training_args.num_train_epochs
    )

    # Initialize Trainer with optimizer and scheduler
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        optimizers=(optimizer, scheduler),
        # Remove EarlyStoppingCallback
    )

    # Train the model
    trainer.train()
    
    # Clear cache to free up memory
    torch.cuda.empty_cache()

    # Optionally, save the model for this fold
    trainer.save_model(f"./trained_model_fold_{fold + 1}")

# After k-fold training, you might want to select the best model
# For simplicity, we'll use the last trained model, but you could implement a selection process based on validation performance

# Save the tokenizer and final model 
tokenizer.save_pretrained("./trained_model")
model.save_pretrained("./trained_model")

# Save the intent to id mapping
with open('./trained_model/label_mappings.json', 'w') as f:
    json.dump(intent_to_id, f)

print("Training completed. Model and tokenizer saved.")
