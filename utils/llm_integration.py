import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.extract_local_context import extract_local_context, represent_local_context  # Import functions from utils
import numpy as np
from dotenv import load_dotenv
import os
import re

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
        outputs = self.llm.generate(**inputs, do_sample=True, max_new_tokens = 10, temperature=0.1) # Reduce max_length since its generating less data now
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print("LLM response: ", response)
        return response

    def extract_context_value(self,response):
        """
        Extracts the context value from the LLM response.

        Args:
            response (str): The response from the LLM.

        Returns:
            float: The context value as a floating-point number.
                Returns None if parsing fails.
        """
        match = re.search(r'Context Value:\s*(-?\d+(\.\d+)?)', response)
        if match:
            return float(match.group(1))  # Extract and convert to float
        else:
            # print("⚠️ Error: Context value +0 found in LLM response")
            return None  # Handle missing values gracefully

    def create_prompt(self, state, action, reward, next_state, done, maze):
        """
        Creates a prompt for the LLM focusing on generating an informed context value.

        Args:
            state (str): Current state representation.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (str): Next state representation.
            done (bool): Whether the episode is done.
            maze (np.array): The maze environment.

        Returns:
            str: A prompt to send to the LLM.
        """
        agent_position = self.decode_state(str(state))


        maze = np.array(maze)
        # print("Maze:", maze)

        local_context = extract_local_context(maze, agent_position)
        local_context_str = represent_local_context(local_context)
        # print(f"Local context for agent at position: {agent_position}")
        # print(local_context_str)

        # prompt = f"Local Context:\n{local_context_str}\n"
        # prompt += f"State: {state}\n"
        # prompt += f"Action: {action}\n"
        # prompt += f"Reward: {reward}\n"
        # prompt += f"Next State: {next_state}\n"
        # prompt += f"Done: {done}\n"
        # prompt += "Based on this information, provide a context value that can be used to improve the Q-value estimation."
        # prompt += "The context value should help refine the action selection process."  # Added context about the role of the context value
        # prompt += " The valid value of the context value should only be between -1 and 1"
        # prompt += " Ensure that you the context value generated by the LLM takes into consideration of the local and global maze context"
        # prompt += "Format: Context Value: [value]"
        prompt = f"""
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
                
                Using these factors, generate a single numerical "context value" that adjusts the Q-value estimation. This value should reflect how favorable or unfavorable the combined local and global contexts are for the current action, where:
                - -1 means highly unfavorable,
                - 0 means neutral, and
                - 1 means highly favorable.
                
                Using these factors, generate a single numerical "context value" that adjusts the Q-value estimation. The output should be exactly in this format:

                Context Value: [value]
                Do not add any explanations, commentary, or extra text. Only return the value in the specified format.
                """

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
