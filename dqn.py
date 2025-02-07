import torch
import numpy as np
from collections import deque
import json
import csv
from torch import nn
import torch.nn.functional as F
import os

# Utility functions
def log_training_data(file_name, episode, steps, reward, epsilon, td_error):
    """Log training data to a CSV file."""
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(["Episode", "Steps", "Total Reward", "Epsilon", "TD Error"])
        writer.writerow([episode, steps, reward, epsilon, td_error])

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]

# DQN Network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.layer1 = nn.Linear(state_size, 32)
        self.layer2 = nn.Linear(32, 16)
        self.layer3 = nn.Linear(16, action_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class MazeEnvironment:
    def __init__(self, maze, maze_size, start, destination):
        self.maze = maze
        self.maze_size = maze_size
        self.start = start
        self.destination = destination
        self.visited_states = []
        self.reset()

    def reset(self):
        self.state = self.start  # Start position
        self.path = [self.start]  # Initialize the path for this episode
        self.steps = 0
        return self.state

    def step(self, action):
        self.steps += 1
        row, col = self.state
        
        if action == 1:  # Right
            new_col = col + 1
            new_row = row
        elif action == 2:  # Down
            new_row = row + 1
            new_col = col
        elif action == 3:  # Left
            new_col = col - 1
            new_row = row
        elif action == 4:  # Up
            new_row = row - 1
            new_col = col
        else:  # No action or invalid action
            new_row, new_col = row, col

        next_state = (new_row, new_col)
        if self.maze[new_row][new_col] == 1:
            done = True
        else:
            done = next_state == self.destination

        reward = self.calculate_reward(self.state, next_state, done, self.steps)

        self.state = next_state
        self.path.append(next_state)  # Add the state to the path
        return next_state, reward, done
    
    def calculate_reward(self, state, next_state, done, step_count):
        """
        Calculate reward based on state transitions.
        Penalizes collisions, rewards reaching the goal, adds step penalties, 
        and increases penalties after 30 steps to encourage faster solutions.
        """
        destination = np.array(self.destination)
        state_arr = np.array(state)
        next_state_arr = np.array(next_state)

        if done:
            if next_state == self.destination:
                print("Destination reached!")
                return 5  # Larger reward for reaching the goal
            else:
                return -0.75  # Larger penalty for a collision or invalid termination

        # Base reward for valid steps
        reward = 0.05

        # Penalize revisiting states to encourage exploration
        if next_state not in self.visited_states:
            reward += 0.05
            self.visited_states.append(next_state)
        else:
            reward -= 0.07

        # Distance-based reward
        current_dist = np.linalg.norm(destination - state_arr)
        next_dist = np.linalg.norm(destination - next_state_arr)
        if next_dist < current_dist:
            reward += 0.05  # Reward getting closer to the goal
        else:
            reward -= 0.1  # Penalize moving farther from the goal

        return reward

# DQN Model Wrapper
class DQNModel:
    def __init__(self, state_size, action_size, maze, device):
        param_dir = 'Simulation/Utils/'
        with open(param_dir + 'config.json', 'r') as file:
            self.params = json.load(file)

        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.maze = maze

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

    def encode_state(self, state):
        one_hot_tensor = torch.zeros(self.state_size, dtype=torch.float32, device=self.device)
        row, col = state
        state_index = row * self.maze.maze_size + col
        one_hot_tensor[state_index] = 1
        return one_hot_tensor

    def update_target_network(self):
        self.target_network.load_state_dict(self.main_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
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

        for state, action, reward, next_state, done in minibatch:
            state_tensor = self.encode_state(state).to(self.device)
            next_state_tensor = self.encode_state(next_state).to(self.device)
            reward_tensor = torch.tensor([reward], device=self.device)

            with torch.no_grad():
                target = reward_tensor + (0 if done else self.gamma * self.target_network(next_state_tensor).max())
            target_Q = self.main_network(state_tensor).clone()
            target_Q[action] = target
            predicted_Q = self.main_network(state_tensor)
            predicted_Q_values.append(predicted_Q)
            target_Q_values.append(target_Q)

        loss = self.loss_fn(torch.stack(predicted_Q_values), torch.stack(target_Q_values))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

# Training Loop
def run_training_loop(dqn_model, maze_env, num_episodes, max_steps, batch_size, target_update_freq):
    total_rewards = []
    file_path = "training_log.csv"
    model_save_path = "model_final_trained.pt"  # Path to save the model
    best_reward = -float("inf")
    destination_count = 0

    for episode in range(num_episodes):
        state = maze_env.reset()
        total_reward = 0
        td_error_sum = 0

        for step in range(max_steps):
            action = dqn_model.act(state)
            next_state, reward, done = maze_env.step(action)
            dqn_model.remember(state, action, reward, next_state, done)

            if len(dqn_model.replay_buffer.buffer) >= batch_size:
                td_error_sum += dqn_model.train(batch_size).item()

            state = next_state
            total_reward += reward
            if done:
                if next_state == maze_env.destination:
                    destination_count += 1
                    print(f"=== Destination Count : {destination_count} ===")
                break

        if episode % target_update_freq == 0:
            dqn_model.update_target_network()

        dqn_model.epsilon = max(dqn_model.epsilon * dqn_model.epsilon_decay, dqn_model.epsilon_min)
        total_rewards.append(total_reward)
        avg_td_error = td_error_sum / max(1, step + 1)
        log_training_data(file_path, episode, step + 1, total_reward, dqn_model.epsilon, avg_td_error)
        
        # Print path taken by the agent
        print(f"Episode {episode + 1}: Path: {maze_env.path}")
        print(f"Total Reward: {total_reward}, Steps: {step+1}, Avg TD Error: {avg_td_error} \n")

        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(dqn_model.main_network.state_dict(), model_save_path)
            print(f"New best reward {best_reward} achieved. Model saved at {model_save_path}.")

    print(f"Model saved to {model_save_path}")
    return total_rewards

def test_model(dqn_model, num_episodes, max_steps, model_load_path="model_final_trained.pt"):
    checkpoint = torch.load(model_load_path, weights_only=True)
    dqn_model.main_network.load_state_dict(checkpoint)  # Load the trained model
    dqn_model.main_network.eval()  # Set model to evaluation mode
    dqn_model.epsilon = 0
    print(f"Loaded model from {model_load_path} for testing.")

    total_rewards = []
    for episode in range(num_episodes):
        state = dqn_model.maze.reset()  # Reset environment within the model
        total_reward = 0

        for step in range(max_steps):
            action = dqn_model.act(state)  # Greedy action selection
            next_state, reward, done = dqn_model.maze.step(action)
            total_reward += reward
            print(f"Step {step+1}: Move from {state} to {next_state}. Reward: {reward}")
            state = next_state
            if done:
                break

        total_rewards.append(total_reward)
        print(f"Total Reward: {total_reward}")

    print(f"Testing complete. Average reward over {num_episodes} episodes: {sum(total_rewards) / num_episodes}")
    return total_rewards

if __name__ == "__main__":
    print("Running")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maze = [
                        [1,1,1,1,1,1,1,1,1,1],
                        [1,1,0,0,0,0,0,0,0,1],
                        [1,0,0,0,0,0,0,0,0,1],
                        [1,0,0,0,0,0,1,0,1,1],
                        [1,0,1,0,0,1,1,0,0,1],
                        [1,0,1,0,0,0,0,0,1,1],
                        [1,1,0,0,0,0,0,0,1,1],
                        [1,0,0,1,0,1,0,0,0,1],
                        [1,1,0,0,0,1,0,1,0,1],
                        [1,1,1,1,1,1,1,1,1,1],
                    ]
    maze_env = MazeEnvironment(maze=maze, maze_size=10, start=(1,6), destination=(8,8))
    state_size = maze_env.maze_size * maze_env.maze_size
    action_size = 5  # Example: none, right, down, left, up
    dqn_model = DQNModel(state_size, action_size, maze_env, device)
    run_training_loop(dqn_model, maze_env, num_episodes=3000, max_steps=100, batch_size=32, target_update_freq=10)
    # start_time = time.time()
    # test_model(dqn_model=dqn_model,num_episodes=100,max_steps=75,model_load_path="model_final_trained.pt")
    # inspect_pt_file(model_path="model_final_trained.pt", model_class=DQN, save_weights=True, layer_name="layer1.weight")

