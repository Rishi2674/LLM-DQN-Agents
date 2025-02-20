import torch
from envs.maze_env import MazeEnvironment
from models.dqn_model import DQNModel
from train_model import run_training_loop
from models.dqn_llm_model import DQN_LLM_Model

if __name__ == "__main__":
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
    # dqn_model = DQNModel(state_size=100, action_size=5, maze=maze_env, device=device)
    dqn_model = DQN_LLM_Model(state_size=100,action_size=5,maze=maze_env,device=device )
    print("DQN Model initialized and ready for training/testing.")
    run_training_loop(dqn_model, maze_env, num_episodes=3000, max_steps=100, batch_size=32, target_update_freq=10)