import torch
from utils.logging_utils import log_training_data

def run_training_loop(dqn_model, maze_env, num_episodes, max_steps, batch_size, target_update_freq):
    total_rewards = []
    file_path = "training_log_std.csv"
    model_save_path = "model_final_trained_std.pt"  # Path to save the model
    best_reward = -float("inf")
    destination_count = 0

    for episode in range(num_episodes):
        state = maze_env.reset()
        total_reward = 0
        td_error_sum = 0

        for step in range(max_steps):
            action = dqn_model.act(state)
            next_state, reward, done = maze_env.step(action)
            """Insert LLM Here"""
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