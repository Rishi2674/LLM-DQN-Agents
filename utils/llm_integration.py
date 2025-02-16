import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.extract_local_context import extract_local_context, represent_local_context  # Import functions from utils
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

ACCESS_TOKEN = os.getenv('ACCESS_TOKEN')

class LLMExperienceGenerator:
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf", device='cpu'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=ACCESS_TOKEN, legacy = True)
        self.llm = AutoModelForCausalLM.from_pretrained(model_name, token=ACCESS_TOKEN)
        self.device = device

    def generate_experience(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.llm.generate(**inputs, do_sample=True, max_length=150) # Reduce max_length since its generating less data now
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def parse_llm_response(self, response):
        try:
            parts = response.split(",")
            state = parts[0].split("State: ")[1].strip()
            action = int(parts[1].split("Action: ")[1].strip())
            reward = float(parts[2].split("Reward: ")[1].strip())
            # next_state = parts[3].split("Next State: ")[1].strip()
            done = bool(int(parts[3].split("Done: ")[1].strip()))
            context_value = float(parts[4].split("Context Value: ")[1].strip())
            return state, action, reward, done, context_value
        except (IndexError, ValueError) as e:
            print(f"Error parsing LLM response: {e}, Response was: {response}")
            return None

    def create_prompt(self, state, action, reward, next_state, done, maze):
        agent_position = self.decode_state(state)  # Assuming you have decode_state
        local_context = extract_local_context(maze, agent_position)  # Use the imported function
        local_context_str = represent_local_context(local_context)  # Use the imported function

        prompt = f"Local Context:\n{local_context_str}\n"
        prompt += f"State: {state}\n"
        prompt += f"Action: {action}\n"
        prompt += f"Reward: {reward}\n"
        prompt += f"Next State: {next_state}\n"
        prompt += f"Done: {done}\n"
        prompt += "Based on this information, predict the next state, action, reward and if it's done and a context value."  # Change prompt here.

        prompt += "Format: State: [next_state]; Action: [action]; Reward: [reward]; Done: [0 or 1]; Context Value: [value]" # Adjust the format here.

        return prompt

    def decode_state(self, state_string):
        try:
            row_str, col_str = state_string[1:-1].split(',')
            row = int(row_str.strip())
            col = int(col_str.strip())
            return (row, col)
        except ValueError:
            print(f"Invalid state string format: {state_string}")
            return None
