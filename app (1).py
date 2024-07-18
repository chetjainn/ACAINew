from flask import Flask, request, jsonify
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from generation_Azure import CustomDataset, LanguageGenerationModel  # Assuming this imports your custom classes correctly
import torch
import json

# Create the Flask app
app = Flask(__name__)

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('./saved_models/my_model/')
model = GPT2LMHeadModel.from_pretrained('./saved_models/my_model/')
language_model = LanguageGenerationModel(model_type='gpt2', model_name_or_path='./saved_models/my_model/', tokenizer=tokenizer)

# Example storage for prompts (replace with your actual storage mechanism)
stored_prompts = []

# Endpoint to receive prompt from the user
@app.route('/prompt', methods=['GET', 'POST', 'PUT'])
def handle_prompt():
    if request.method == 'GET':
        return jsonify({"message": "Use POST or PUT to send a prompt"}), 200

    elif request.method == 'POST' or request.method == 'PUT':
        try:
            data = request.get_json(force=True)
            prompt = data.get('prompt', 'No prompt provided')

            # Validate required fields
            if not prompt:
                return jsonify({"error": "No prompt provided in the request"}), 400

            # Generate text using the language model
            generated_texts = language_model.generate(prompt=prompt, verbose=True)
            
            return jsonify({
                "generated_texts": generated_texts
            })

        except Exception as e:
            return jsonify({"error": f"Failed to parse JSON or process request: {str(e)}"}), 400

    else:
        return jsonify({"message": f"Method {request.method} not supported for /prompt"}), 405

# Endpoint for Text Generation
@app.route('/generate', methods=['POST', 'PUT'])
def handle_generate():
    if request.method == 'POST' or request.method == 'PUT':
        try:
            data = request.get_json(force=True)
            prompt = data.get('prompt', 'No prompt provided')
            verbose = data.get('verbose', True)  # Optional: verbose flag for printing generated text

            # Validate required fields
            if not prompt:
                return jsonify({"error": "No prompt provided in the request"}), 400

            # Generate text using the language model
            generated_texts = language_model.generate(prompt=prompt, verbose=verbose)

            return jsonify({"generated_texts": generated_texts})

        except Exception as e:
            return jsonify({"error": f"Failed to parse JSON or process request: {str(e)}"}), 400
    else:
        return jsonify({"message": f"Method {request.method} not supported for /generate"}), 405

# Endpoint for Model Training (Optional)
@app.route('/train', methods=['POST', 'PUT'])
def handle_train():
    # Load dataset from Azure Blob storage
    default_data_file = 'https://acailanguage.blob.core.windows.net/testdata/data.json'
    
    if request.method == 'POST' or request.method == 'PUT':
        try:
            data = request.get_json(force=True)
            data_file = data.get('data_file', default_data_file)
            num_epochs = data.get('num_epochs', 3)
            batch_size = data.get('batch_size', 4)

            # Validate required fields
            if not data_file:
                return jsonify({"error": "No data_file provided in the request"}), 400

            # Initialize optimizer and scheduler if not already initialized
            if language_model.optimizer is None:
                language_model.optimizer = AdamW(language_model.model.parameters(), lr=2e-5)
                num_training_steps = num_epochs * len(train_dataset) // batch_size
                language_model.scheduler = get_linear_schedule_with_warmup(
                    language_model.optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
                )

            # Assuming CustomDataset is defined properly and tokenizer is accessible
            train_dataset = CustomDataset(data_file, language_model.tokenizer)

            # Call train method with optimizer and scheduler
            language_model.train(train_dataset, num_epochs=num_epochs, batch_size=batch_size)

            # Save model checkpoint
            language_model.save_model_args('./saved_models/my_model/')

            return jsonify({"message": "Model trained and checkpoint saved successfully"})

        except Exception as e:
            return jsonify({"error": f"Failed to parse JSON or process request: {str(e)}"}), 400
    else:
        return jsonify({"message": f"Method {request.method} not supported for /train"}), 405
# Handle requests for undefined routes explicitly
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def handle_undefined_routes(path):
    return jsonify({"error": "Route not found. Please check your URL."}), 404

# Suppress the warning for FutureWarning in optimization.py
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.optimization")

# Running the Flask app
if __name__ == '__main__':
    app.run(debug=True)



