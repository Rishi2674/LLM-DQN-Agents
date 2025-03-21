import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.extract_local_context import extract_local_context, represent_local_context  # Import functions from utils
import numpy as np
from dotenv import load_dotenv
import os
import re
from typing import Optional, List
import json

load_dotenv()

ACCESS_TOKEN = os.getenv('ACCESS_TOKEN')

class LLMExperienceGenerator:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.3", device='cuda'):
        # print("Before model loading:")
        # print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        # print(f"Cached: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
        # for obj in gc.get_objects():
        #     if isinstance(obj, torch.nn.Module):
        #         print(f"Model: {type(obj).__name__}, Device: {next(obj.parameters()).device}")

            # total_mem = 0
            # for obj in gc.get_objects():
            #     if isinstance(obj, torch.Tensor) and obj.is_cuda:
            #         mem = obj.element_size() * obj.numel() / (1024 ** 2)  # Convert bytes to MB
            #         total_mem += mem
            #         print(f"Tensor: {obj.shape}, Device: {obj.device}, Memory: {mem:.2f} MB")

        # print(f"Total Tensor Memory Usage: {total_mem:.2f} MB")
        torch.cuda.empty_cache()
        print(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=ACCESS_TOKEN, legacy = True)
        self.llm = AutoModelForCausalLM.from_pretrained(model_name, token=ACCESS_TOKEN, load_in_4bit = True, device_map = "auto")
        # for obj in torch.cuda.memory_allocated(), torch.cuda.memory_reserved():
        #     print(f"CUDA Memory Usage: {obj / (1024 ** 3):.2f} GiB")
        # self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # print("till here")
        # self.llm.to(self.device)
        # print("maybe issue here")

        # print("After model loading:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
        # print(f"Model is on: {next(self.llm.parameters()).device}")


    def generate_experience(self, prompt):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        # print(f"Cached: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
        torch.cuda.empty_cache()
        torch.cuda.memory_allocated()
        # self.llm.to(device)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        outputs = self.llm.generate(**inputs, do_sample=False, max_new_tokens = 50, temperature=0.1) # Reduce max_length since its generating less data now
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print("LLM response: ", response)
        return response

    def extract_context_vector(self,llm_response: str, expected_length: int = 5) -> Optional[List[float]]:
        # Attempt JSON parsing
        try:
            data = json.loads(llm_response.strip())
            if "context_values" in data:
                values = data["context_values"]
                if isinstance(values, list) and len(values) == expected_length:
                    # Ensure all values are floats and clamp between -1 and 1
                    return [max(-1, min(1, float(v))) for v in values]
        except Exception:
            pass  # Fall back to regex extraction if JSON parsing fails

        # Fallback: use regex to find JSON-like pattern
        regex_pattern = r'\{"context_values":\s*\[([^\]]+)\]\}'
        match = re.search(regex_pattern, llm_response)
        if match:
            values_str = match.group(1)
            try:
                # Split by commas and convert to float
                values = [float(x.strip()) for x in values_str.split(",")]
                if len(values) == expected_length:
                    return [max(-1, min(1, v)) for v in values]
            except Exception as e:
                print("Error converting extracted values to floats:", e)
                return None

        # If extraction fails, return None
        print("Context vector extraction failed. LLM response was:")
        print(llm_response)
        return None


    def create_prompt(self, state, action, reward, next_state, done, maze):

        agent_position = self.decode_state(str(state))


        maze = np.array(maze)
        # print("Maze:", maze)

        local_context = extract_local_context(maze, agent_position)
        local_context_str = represent_local_context(local_context)

        # prompt = f"""
        #         Local Context: {local_context_str}
        #         State: {state}
        #         Action: {action}
        #         Reward: {reward}
        #         Next State: {next_state}
        #         Done: {done}
        #
        #         The Q-value of an action represents the expected future reward. However, local and global maze contexts can modify this estimate. Consider the following factors:
        #         1. Local Maze Structure: The arrangement of obstacles (represented by 1's) and clear paths (represented by 0's) in the provided grid indicates how navigable the immediate area is.
        #         2. Global Maze Influence: Although not fully detailed here, assume that the overall maze layout impacts the effectiveness of an action. For instance, a clear local context might be less advantageous if it leads to a dead end globally.
        #         3. Reward Signal: The immediate reward indicates the benefit (or cost) of the action taken.
        #         4. State Transition: The change from the current state to the next state can signal progress toward a goal or potential pitfalls.
        #
        #         Using these factors, generate a single numerical "context value" that adjusts the Q-value estimation. This value should reflect how favorable or unfavorable the combined local and global contexts are for the current action, where:
        #         - -1 means highly unfavorable,
        #         - 0 means neutral, and
        #         - 1 means highly favorable.
        #
        #         Using these factors, generate a single numerical "context value" that adjusts the Q-value estimation. The output should be exactly in this format:
        #
        #         Context Value: [value]
        #         Do not add any explanations, commentary, or extra text. Only return the value in the specified format.
        #         """


        prompt = f"""[INST]
            Local Context: {local_context_str}
            State: {state}
            Action: {action}
            Reward: {reward}
            Next State: {next_state}
            Done: {done}
            
            The Q-value of an action represents the expected future reward. However, local and global maze contexts can modify this estimate. Consider the following factors:
            1. Local Maze Structure: The arrangement of obstacles (represented by 1's) and clear paths (represented by 0's) in the provided grid indicates how navigable the immediate area is.
            2. Global Maze Influence: Although not fully detailed here, assume that the overall maze layout impacts the effectiveness of an action. For instance, a clear local context might be less advantageous if it leads to a dead end globally.
            3. Reward Signal: The immediate reward indicates the benefit (or cost) of the action taken.
            4. State Transition: The change from the current state to the next state can signal progress toward a goal or potential pitfalls.
            
            Using these factors, generate a context vector that adjusts the Q-value estimation for each action.
            Your output must be **strictly** in the following JSON format:
            
            {{"context_values": [v1, v2, v3, v4]}}
            
            Where:
            - `v1` corresponds to moving **right**.
            - `v2` corresponds to moving **down**.
            - `v3` corresponds to moving **left**.
            - `v4` corresponds to moving **up**.
            
            Each value must be a decimal between **-1 and 1**.  
            Do not include explanations or additional text. Only return the JSON object in the specified format.
            
            [/INST]"""


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
