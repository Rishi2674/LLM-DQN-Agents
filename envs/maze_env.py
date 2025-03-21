import numpy as np

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
        
        if action == 0:  # Right
            new_col = col + 1
            new_row = row
        elif action == 1:  # Down
            new_row = row + 1
            new_col = col
        elif action == 2:  # Left
            new_col = col - 1
            new_row = row
        elif action == 3:  # Up
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