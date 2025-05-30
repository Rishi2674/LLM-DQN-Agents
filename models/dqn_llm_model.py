import json
from utils.replay_buffer import ReplayBuffer
from models.neural_network import DQN
import torch
from torch import nn
import numpy as np
from utils.llm_integration import LLMExperienceGenerator,extract_context_vector


class DQN_LLM_Model:
    def __init__(self, state_size, action_size, maze, device):
        param_dir = 'utils/'
        with open(param_dir + 'config.json', 'r') as file:
            self.params = json.load(file)

        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.maze = maze
        # self.maze is actually an instance of the maze environment

        self.buffer_size = self.params['BUFFER_SIZE']
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.gamma = self.params['GAMMA']
        self.alpha = self.params['ALPHA']
        self.epsilon = self.params['EPSILON']
        self.epsilon_min = self.params['EPSILON_MIN']
        self.epsilon_decay = self.params['EPSILON_DECAY']

        self.main_network = DQN(self.state_size, self.action_size).to(device=device)
        self.target_network = DQN(self.state_size, self.action_size).to(device=device)
        self.update_target_network()

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.main_network.parameters(), lr=self.alpha)
        self.llm_generator = LLMExperienceGenerator(device=self.device)

    def encode_state(self, state):
        one_hot_tensor = torch.zeros(self.state_size, dtype=torch.float32, device=self.device)
        row, col = state
        state_index = row * self.maze.maze_size + col
        one_hot_tensor[state_index] = 1
        return one_hot_tensor

    def update_target_network(self):
        self.target_network.load_state_dict(self.main_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """
        Insert LLM Here
        """
        self.replay_buffer.append((state, action, reward, next_state, done))

    def calculate_reward(self, state, next_state, done):
        """
        Calculate reward based on state transitions.
        Penalizes collisions with walls, rewards reaching the goal, and adds a small step penalty.
        """
        if done:
            if next_state == self.maze.destination:
                print("Destination reached!")
                return 5
            else:
                return -0.75
        else:
            reward = 0.05
            if next_state not in self.maze.visited_states:
                reward += 0.05
                self.maze.visited_states.append(next_state)
            else:
                reward -= 0.07
            state_arr = np.array(state)
            next_state_arr = np.array(next_state)
            destination = np.array((8, 8))
            current_dist = np.linalg.norm(destination - state_arr)
            prev_dist = np.linalg.norm(destination - next_state_arr)
            if current_dist <= prev_dist:
                reward += 0.05
            else:
                reward -= 0.10
            return reward

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        state_tensor = self.encode_state(state).to(self.device)
        with torch.no_grad():
            return self.main_network(state_tensor).argmax().item()

    def train(self, batch_size):
        minibatch = self.replay_buffer.sample(batch_size)
        predicted_Q_values, target_Q_values = [], []

        # Process each experience in the minibatch
        for state, action, reward, next_state, done in minibatch:
            # Generate prompt and get LLM response for context vector
            prompt = self.llm_generator.create_prompt(state, action, reward, next_state, done, self.maze.maze)
            llm_response = self.llm_generator.generate_experience(prompt)
            # print("LLM response: ", llm_response)
            context_vector = extract_context_vector(llm_response)
            print(f"State : {state}, Action : {action}, Next State : {next_state}, Reward : {reward}")
            print("context vector:", context_vector)

            # If extraction fails, default to a zero vector of appropriate size
            if context_vector is None:
                context_vector = torch.zeros(self.action_size, dtype=torch.float32, device=self.device)
            else:
                # Convert extracted list to a torch tensor on the proper device
                context_vector = torch.tensor(context_vector, dtype=torch.float32, device=self.device)

            # Encode current and next states as tensors
            state_tensor = self.encode_state(state).to(self.device)
            next_state_tensor = self.encode_state(next_state).to(self.device)
            reward_tensor = torch.tensor([reward], device=self.device)

            # Compute target Q value using the target network and Bellman equation
            with torch.no_grad():
                target = reward_tensor + (0 if done else self.gamma * self.target_network(next_state_tensor).max())
            # Get a copy of predicted Q-values and update the value for the taken action with the target
            target_Q = self.main_network(state_tensor).clone()
            target_Q[action] = target

            # Obtain predicted Q-values from the main network
            predicted_Q = self.main_network(state_tensor)
            # Apply the context vector (element-wise addition) to the predicted Q-values

            #the Q value has increased here, but that would lead to increase in getting Q value getting closer to the target Q value
            # which might reduce the loss and cause gradient updates which are not very enough for the model to converge.
            # solutions??
            predicted_Q = predicted_Q + context_vector

            predicted_Q_values.append(predicted_Q)
            target_Q_values.append(target_Q)

        loss = self.loss_fn(torch.stack(predicted_Q_values), torch.stack(target_Q_values))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
