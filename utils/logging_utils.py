import csv

def log_training_data(file_name, episode, steps, reward, epsilon, td_error):
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(["Episode", "Steps", "Total Reward", "Epsilon", "TD Error"])
        writer.writerow([episode, steps, reward, epsilon, td_error])