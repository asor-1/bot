import os
import sys
import torch
import logging
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from setup import setup_rag, setup_intent_classification
from functions.osm_OpenMind_function import final_Directions

app = Flask(__name__)
CORS(app)

# Global variables
rag = None
tokenizer = None
model = None
device = None

def init_resources():
    global rag, classifier, id_to_intent
    json_file_path = os.path.abspath(r"C:\Users\adida\Desktop\Assistance_Bot\model\knowledge_base.json")
    vectorstore_path = os.path.abspath(os.path.join("model", "vectors"))
    rag = setup_rag(json_file_path, vectorstore_path)
    classifier, id_to_intent = setup_intent_classification("./trained_model", confidence_threshold=0.4)

def handle_exception(exc_type, exc_value, exc_traceback):
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception
# Define intent_to_id mapping (replace with your actual mapping)
intent_to_id = {
    "Capital": 0,
    "directions": 1,
}

@app.route('/')
def index():
    return render_template('index.html')

CONFIDENCE_THRESHOLD = 0.5

@app.route('/chat', methods=['POST'])
def chat():
    user_query = request.json.get('query', '')
    if not user_query:
        logging.error("No query provided in the request")
        return jsonify({"error": "No query provided"}), 400

    try:
        logging.info(f"Processing query: {user_query}")

        # Intent classification
        predicted_class, confidence = classifier.classify(user_query)
        
        if classifier.is_confident(confidence):
            intent = id_to_intent[predicted_class]
        else:
            intent = "unknown"
            # Fallback mechanism
            if confidence > 0.4:  # Still somewhat confident
                logging.info(f"Low confidence prediction: {id_to_intent[predicted_class]} (confidence: {confidence:.2f})")
                # You could use this low confidence prediction or ask for clarification
            else:
                logging.info("Very low confidence, using general response")
            # Use a general response or ask the user for clarification

        logging.info(f"Classified intent: {intent} (confidence: {confidence:.2f})")

        if intent == "directions":
            # Call your function for directions here
            answer = final_Directions(41.8789, -87.6359)  # Example coordinates, replace with actual logic
            response = {
                "intent": intent,
                "answer": answer
            }
        else:
            result = rag.query(user_query)
            response = {
                "intent": intent,
                "answer": result['answer'],
                "sources": result.get('sources', [])
            }

        logging.info("Query processed successfully")
        return jsonify(response)

    except Exception as e:
        logging.exception(f"Error processing query: {str(e)}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    init_resources()
    app.run(debug=False)  # Set to True for development, False for troubleshooting
