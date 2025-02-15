import torch


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