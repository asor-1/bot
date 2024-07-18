import os
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from model_rag import AdvancedRAG
import torch
import torch.nn.functional as F
import json

# RAG setup
def setup_rag(json_file_path, vectorstore_path):
    rag = AdvancedRAG(json_file_path)

    # Checking to see if we already have a vectorstore if we have a saved vectorstore
    index_file = os.path.join(vectorstore_path, "index.faiss")
    if os.path.exists(index_file):
        print(f"Loading existing vectorstore from {index_file}")
        rag.load_vectorstore(vectorstore_path)
    else:
        print(f"Vectorstore not found at {index_file}. Creating new vectorstore.")
        rag.setup()
        os.makedirs(vectorstore_path, exist_ok=True)
        rag.save_vectorstore(vectorstore_path)

    return rag

# Intent classification setup
class IntentClassifier:
    def __init__(self, model, tokenizer, device, confidence_threshold=0.4):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.confidence_threshold = confidence_threshold

    def classify(self, text):
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()

        return predicted_class, confidence

    def is_confident(self, confidence):
        return confidence >= self.confidence_threshold

def setup_intent_classification(model_path, confidence_threshold=0.4):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Ensure model is in evaluation mode

    classifier = IntentClassifier(model, tokenizer, device, confidence_threshold)

    # Load intent to id mapping
    with open(os.path.join(model_path, 'label_mappings.json'), 'r') as f:
        intent_to_id = json.load(f)
    id_to_intent = {v: k for k, v in intent_to_id.items()}

    return classifier, id_to_intent