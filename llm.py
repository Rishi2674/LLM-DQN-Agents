from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import os

load_dotenv()

ACCESS_TOKEN = os.getenv('ACCESS_TOKEN')  # Get the token from your .env file
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
# Print the access token to ensure it's being passed correctly
print(f"Using access token: {ACCESS_TOKEN}")

# Try loading the model and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=ACCESS_TOKEN, legacy = True)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")

try:
    model = AutoModelForCausalLM.from_pretrained(model_name, token=ACCESS_TOKEN)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Generate text
prompt = """
    What is the capital Of indiB?
    Strictly answer in one word
"""
inputs = tokenizer(prompt, return_tensors="pt")  # Move to GPU if available
output = model.generate(**inputs, max_new_tokens = 50, temperature=0.1, top_p=0.9, repetition_penalty=1.2)

# Print response
print(tokenizer.decode(output[0], skip_special_tokens=True))
