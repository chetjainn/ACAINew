import json
import logging
import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

# Define your preprocessing functions if needed
PREPROCESSING_FUNCTIONS = {}

from azure.storage.blob import BlobServiceClient

logger = logging.getLogger(__name__)

class CustomDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=512):
        self.data = self.load_data_from_azure(data_file)  # Update to use Azure loader
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Set padding token if not already set
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.pad_token = '[PAD]'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        response = item["response"]
        
        # Preprocess prompt and response to handle special tokens
        inputs = self.preprocess_text(prompt, response)
        
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }

        prompt = item["prompt"]
        response = item["response"]
        
        # Preprocess prompt and response to handle special tokens
        inputs = self.preprocess_text(prompt, response)
        
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }

    def preprocess_text(self, prompt, response):
        # Concatenate prompt and response
        text = f"{prompt} {response}"
        
        # Define a regex pattern to identify special characters you want to replace
        special_char_pattern = r'[^\w\s]'  # Example: Replace any non-alphanumeric characters
        
        # Replace special characters with a space or any other suitable replacement
        text = re.sub(special_char_pattern, ' ', text)
        
        # Tokenize the text
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Validate input_ids against tokenizer's vocab size
        input_ids = inputs.input_ids.squeeze()
        if (input_ids >= self.tokenizer.vocab_size).any():
            # Instead of raising an error, handle this case by truncating or special token replacement
            # For demonstration, we'll truncate the input_ids if they exceed vocab size
            input_ids = input_ids.masked_fill(input_ids >= self.tokenizer.vocab_size, self.tokenizer.unk_token_id)
        
        return {
            "input_ids": input_ids,
            "attention_mask": inputs.attention_mask.squeeze()
        }

    def load_data_from_azure(self, data_file):
        from azure.storage.blob import BlobServiceClient

        blob_service_client = BlobServiceClient.from_connection_string("DefaultEndpointsProtocol=https;AccountName=acailanguage;AccountKey=lJ+9JPqo/kM7GwoT6HM7zO8URjAm0ZUr0oe1MgHDbC3WtVuMP6gGgXOpb/GcITb6s+QlsMDnqCq1+ASt5RSmsA==;EndpointSuffix=core.windows.net")

        # Parse the URL to extract container name and blob name
        container_name = "testdata"
        blob_name = "data.json"

        container_client = blob_service_client.get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_name)

        # Download blob data
        blob_data = blob_client.download_blob()
        data = json.loads(blob_data.readall().decode('utf-8'))

        return data

class LanguageGenerationModel:
    def __init__(self, model_type, model_name_or_path, tokenizer, args=None, use_cuda=True, **kwargs):
        MODEL_CLASSES = {
            "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
            # Add other model classes as needed
        }

        self.args = args or {}  # Ensure args is at least an empty dict if not provided
        self.model_type = model_type

        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

        model_class, tokenizer_class = MODEL_CLASSES[model_type]

        # Initialize tokenizer and model
        self.tokenizer = tokenizer_class.from_pretrained(model_name_or_path, **kwargs)
        self.model = model_class.from_pretrained(model_name_or_path, **kwargs)
        self.model.to(self.device)

        self.optimizer = None
        self.scheduler = None
        self.num_training_steps = 0
        self.train_dataloader = None  # Initialize train_dataloader attribute

        if self.args.get('do_train', False):
            self._prepare_optimizer_and_scheduler()

    def _prepare_optimizer_and_scheduler(self):
        if not self.args:
            raise ValueError("Training arguments (args) must be provided.")

        self.optimizer = AdamW(self.model.parameters(), lr=self.args.get('learning_rate', 2e-5))
        if self.train_dataloader is not None:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.args.get('warmup_steps', 100),
                num_training_steps=self.num_training_steps,
            )
        else:
            raise ValueError("train_dataloader must be initialized before preparing optimizer and scheduler.")

    def train(self, train_dataset, num_epochs=3, batch_size=4):
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.num_training_steps = len(self.train_dataloader) * num_epochs

        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            for step, batch in enumerate(self.train_dataloader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()

                if (step + 1) % 10 == 0:  # Print average loss every 10 steps
                    avg_loss = total_loss / (step + 1)
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{step + 1}/{len(self.train_dataloader)}], Average Loss: {avg_loss:.4f}')

            avg_train_loss = total_loss / len(self.train_dataloader)
            print(f'Epoch {epoch + 1}/{num_epochs}, Average Training Loss: {avg_train_loss:.4f}')

        # Optionally, save the model after training
        self.save_model_args('./saved_models/my_model/')
        self.model.save_pretrained('./saved_models/my_model/')
        self.tokenizer.save_pretrained('./saved_models/my_model/')
    
    def generate(self, prompt=None, args=None, verbose=True):
        model = self.model
        tokenizer = self.tokenizer
        device = self.device

        if args:
            self.args.update(args)  # Update existing args with new values

        if prompt:
            self.args['prompt'] = prompt
        elif 'prompt' not in self.args:  # Check if prompt attribute exists
            self.args['prompt'] = input("Model prompt >>> ")

        prompt_text = self.args['prompt']

        # Different models need different input formatting and/or extra arguments
        requires_preprocessing = self.model_type in PREPROCESSING_FUNCTIONS.keys()
        if requires_preprocessing:
            prepare_input = PREPROCESSING_FUNCTIONS[self.model_type]
            preprocessed_prompt_text = prepare_input(
                self.args, model, tokenizer, prompt_text
            )
            encoded_prompt = tokenizer.encode(
                preprocessed_prompt_text,
                add_special_tokens=True,
                return_tensors="pt",
                max_length=model.config.max_position_embeddings,  # Access max_position_embeddings from model config
                truncation=True
            ).to(device)
        else:
            encoded_prompt = tokenizer.encode(
                prompt_text,
                add_special_tokens=True,
                return_tensors="pt",
                max_length=model.config.max_position_embeddings,  # Access max_position_embeddings from model config
                truncation=True
            ).to(device)

        # Set pad_token_id for open-end generation
        pad_token_id = tokenizer.eos_token_id
        gen_args = {
            "max_length": self.args.get('max_length', 450),
            "temperature": self.args.get('temperature', 1.0),
            "top_k": self.args.get('top_k', 50),
            "top_p": self.args.get('top_p', 1.0),
            "repetition_penalty": self.args.get('repetition_penalty', 1.0),
            "pad_token_id": pad_token_id,
            "do_sample": True,
            "num_return_sequences": self.args.get('num_return_sequences', 1)
        }

        # Generate text
        output_sequences = model.generate(
            input_ids=encoded_prompt,
            attention_mask=encoded_prompt.ne(pad_token_id),  # Create attention mask excluding padding tokens
            **gen_args
        )

        # Decode generated sequences
        generated_texts = []
        for sequence in output_sequences:
            generated_text = tokenizer.decode(sequence, skip_special_tokens=True)
            generated_texts.append(generated_text)
            if verbose:
                print(generated_text)

        return generated_texts

    def save_model_args(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "training_args.json"), "w") as f:
            json.dump(self.args, f, indent=2)  # Save all args to file

    def get_prompt(self):
        return self.args.get('prompt', 'No prompt set')
