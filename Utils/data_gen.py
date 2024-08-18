import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

def generate_synthetic_data(seq_len: int, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
    x = np.linspace(0, 100, num_samples)
    data = np.sin(x)
    sequences = []
    for i in range(len(data) - seq_len):
        sequences.append(data[i:i + seq_len + 1])
    sequences = np.array(sequences)
    x_data = sequences[:, :-1].reshape(-1, seq_len, 1)
    y_data = sequences[:, -1].reshape(-1, 1)
    return torch.tensor(x_data, dtype=torch.float32), torch.tensor(y_data, dtype=torch.float32)

def simulate_traffic_data(duration=24, time_step=5) -> np.ndarray:
    time_points = duration * (60 // time_step)  # Number of time points in 24 hours
    traffic = np.sin(np.linspace(0, 4 * np.pi, time_points)) + np.random.normal(0, 0.5, time_points)
    traffic = (traffic - traffic.min()) / (traffic.max() - traffic.min())  # Normalize between 0 and 1
    return traffic

def simulate_bandwidth_usage(num_users=100, traffic_len=288) -> np.ndarray:
    traffic = simulate_traffic_data()
    user_usage = np.random.choice(traffic, (num_users, traffic_len))
    return user_usage

# Generate and save the synthetic data
def generate():
    traffic = simulate_traffic_data()
    usage = simulate_bandwidth_usage(traffic_len=len(traffic))

    np.save('../traffic.npy', traffic)
    np.save('../usage.npy', usage)

    print("First 10 data points of traffic data:\n", traffic[:10])
    print("\nFirst 10 rows of user bandwidth usage data:\n", usage[:10])

# Plot the traffic data
    plt.figure(figsize=(10, 5))
    plt.plot(traffic, label='Simulated Network Traffic')
    plt.xlabel('Time Points')
    plt.ylabel('Normalized Traffic')
    plt.title('Simulated Network Traffic Over 24 Hours')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot the first 10 users' bandwidth usage data
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.plot(usage[i], label=f'User {i+1}')
    plt.xlabel('Time Points')
    plt.ylabel('Bandwidth Usage')
    plt.title('Simulated User Bandwidth Usage Over 24 Hours')
    plt.legend()
    plt.grid(True)
    plt.show()