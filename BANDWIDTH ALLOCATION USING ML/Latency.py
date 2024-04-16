import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Genetic Algorithm Parameters
population_size = 50
max_generations = 100
mutation_rate = 0.1

# Traffic Demands and Total Bandwidth
traffic_demands = [20, 30, 40, 10, 5]  # Example traffic demands for users or applications
total_bandwidth = 100

# Initialize Population
def initialize_population(population_size):
    population = []
    for _ in range(population_size):
        chromosome = [random.randint(1, 50) for _ in range(8)]  # Random values for each gene
        population.append(chromosome)
    return population

# Fitness Evaluation
def fitness_evaluation(chromosome):
    Bavail = chromosome[5]  # Free bandwidth available in the cell or neighboring cell
    Bfair = chromosome[4]  # Fair bandwidth allocated for real-time users
    T = chromosome[3]  # Time slot by the real-time user
    N = chromosome[0]  # Number of users generating calls with a particular service
    Pdata = chromosome[2]  # Data packets of real-time users
    latency = (N * Pdata) / (Bavail * T * 100)  # Example latency calculation
    return latency

# Initialize Population
population = initialize_population(population_size)
best_latencies = []

# Main Genetic Algorithm Loop
for generation in range(max_generations):
    # Evaluate Fitness of each chromosome in Population
    latencies = [fitness_evaluation(chromosome) for chromosome in population]
    best_latency = min(latencies)
    best_latencies.append(best_latency)

    # Other genetic algorithm steps...
    # For brevity, the rest of the genetic algorithm code is omitted.

# Store the best chromosome from the genetic algorithm
best_chromosome_idx = latencies.index(min(latencies))
best_chromosome = population[best_chromosome_idx]
ga_latency_allocations = best_chromosome[:5]  # Assuming first 5 genes represent latency allocations

# Generate training data for neural network
np.random.seed(0)
num_samples = 1000
num_features = 10
X = np.random.rand(num_samples, num_features)
y = np.random.rand(num_samples)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train neural network
mlp = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
mlp.fit(X_train_scaled, y_train)

# Evaluate neural network
y_pred = mlp.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Use the trained neural network to predict latency
predicted_latencies = mlp.predict(X_test_scaled)

# Apply bandwidth allocation algorithms
pf_allocation = [(demand / sum(traffic_demands)) * total_bandwidth for demand in traffic_demands]
mm_allocation = [min(total_bandwidth, demand) for demand in traffic_demands]
wfq_allocation = [(weight / sum([1, 2, 3, 1, 2])) * total_bandwidth for weight in [1, 2, 3, 1, 2]]  # Example weights

# Define the category names
categories = ['Throughput', 'Latency', 'Packet Loss Rate', 'Fairness', 'QoS', 'Energy Consumption']

# Predicted results for each category
predicted_results = {
    'Throughput': predicted_latencies[:5],
    'Latency': predicted_latencies[5:10],
    'Packet Loss Rate': predicted_latencies[10:15],
    'Fairness': predicted_latencies[15:20],
    'QoS': predicted_latencies[20:25],
    'Energy Consumption': predicted_latencies[25:]
}

# Print predicted results for each category
for category in categories:
    print("Predicted results for", category)
    print(predicted_results[category])
    print()

# Plotting the comparison graph
plt.figure(figsize=(12, 8))

# Bar width
bar_width = 0.2

# Position of bars on x-axis
r = np.arange(len(traffic_demands))

# Create bars for Genetic Algorithm
plt.bar(r - 1.5*bar_width, ga_latency_allocations, color='purple', width=bar_width, edgecolor='grey', label='Genetic Algorithm')

# Create bars for other algorithms
plt.bar(r - 0.5*bar_width, pf_allocation, color='b', width=bar_width, edgecolor='grey', label='Proportional Fairness')
plt.bar(r + 0.5*bar_width, mm_allocation, color='g', width=bar_width, edgecolor='grey', label='Max-Min Fairness')
plt.bar(r + 1.5*bar_width, wfq_allocation, color='r', width=bar_width, edgecolor='grey', label='Weighted Fair Queuing')

# Add xticks on the middle of the group bars
plt.xlabel('Users/Applications', fontweight='bold')
plt.ylabel('Latency (ms)', fontweight='bold')
plt.xticks(r, [f'User/App {i+1}' for i in range(len(traffic_demands))])

# Create legend & Show graphic
plt.legend()
plt.title('Latency Comparison for Users/Applications')
plt.tight_layout()
plt.show()
