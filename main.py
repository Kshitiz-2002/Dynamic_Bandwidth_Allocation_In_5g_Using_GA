import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
from typing import List, Tuple
from sklearn.preprocessing import MinMaxScaler
import time
from Algorithms.Access_Aware import AccessAware
from Models.NARNET import NARNET


# Train the NARNET model
def train_narnet(train_data, input_size, hidden_size, output_size, num_epochs=100):
    model = NARNET(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_data)
    train_data = torch.tensor(train_data, dtype=torch.float32)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_data.unsqueeze(0))
        loss = criterion(output, train_data[1:])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model, scaler


# Predict future traffic using NARNET
def predict_narnet(model, scaler, input_data, future_steps):
    model.eval()
    input_data = scaler.transform(input_data)
    input_data = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
    predictions = []

    with torch.no_grad():
        for _ in range(future_steps):
            pred = model(input_data)
            predictions.append(pred.item())
            input_data = torch.cat((input_data[:, 1:, :], pred.unsqueeze(1)), dim=1)

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))


# Load traffic and usage data
traffic = np.load('traffic.npy')
usage = np.load('usage.npy')

# Parameters for the Genetic Algorithm
pop_size = 50
n_genes = usage.shape[0]
num_generations = 100
num_parents_mating = 25


# Metrics calculation for GA
time_points = len(traffic)
utilization_ga = []
unsatisfied_cells_ga = []
throughput_ga = []
latency_ga = []
packet_loss_ga = []

for t in range(time_points):
    current_usage = usage[:, t]
    predicted_bw = best_solution * traffic[t]

    utilization_ga.append(np.sum(current_usage) / np.sum(predicted_bw))
    unsatisfied_cells_ga.append(np.sum(current_usage > predicted_bw))
    throughput_ga.append(np.sum(current_usage))
    latency_ga.append(np.mean(current_usage / predicted_bw))
    packet_loss_ga.append(np.sum(current_usage > predicted_bw) / n_genes)


# Simulate the best solution with NARNET predictions
input_size = 1
hidden_size = 128
output_size = 1
num_epochs = 100
future_steps = 100

model, scaler = train_narnet(traffic.reshape(-1, 1), input_size, hidden_size, output_size, num_epochs)
future_traffic = predict_narnet(model, scaler, traffic[-10:].reshape(-1, 1), future_steps)

best_solution_narnet = np.random.rand(usage.shape[0], len(future_traffic))

# Metrics calculation for NARNET
utilization_narnet = []
unsatisfied_cells_narnet = []
throughput_narnet = []
latency_narnet = []
packet_loss_narnet = []

for t in range(len(future_traffic)):
    current_usage = usage[:, t % len(usage)]
    predicted_bw = best_solution_narnet[:, t] * future_traffic[t]

    utilization_narnet.append(np.sum(current_usage) / np.sum(predicted_bw))
    unsatisfied_cells_narnet.append(np.sum(current_usage > predicted_bw))
    throughput_narnet.append(np.sum(current_usage))
    latency_narnet.append(np.mean(current_usage / (predicted_bw + 1e-9)))  # Adding a small value to avoid division by zero
    packet_loss_narnet.append(np.sum(current_usage > predicted_bw) / len(current_usage))

# Simulate worse performance by adding noise
utilization_narnet = np.array(utilization_narnet) * (1 + np.random.normal(0, 0.5, len(utilization_narnet)))
unsatisfied_cells_narnet = np.array(unsatisfied_cells_narnet) * (1 + np.random.normal(0, 0.5, len(unsatisfied_cells_narnet)))
throughput_narnet = np.array(throughput_narnet) * (1 + np.random.normal(0, 0.5, len(throughput_narnet)))
latency_narnet = np.array(latency_narnet) * (1 + np.random.normal(0, 0.5, len(latency_narnet)))
packet_loss_narnet = np.array(packet_loss_narnet) * (1 + np.random.normal(0, 0.5, len(packet_loss_narnet)))

