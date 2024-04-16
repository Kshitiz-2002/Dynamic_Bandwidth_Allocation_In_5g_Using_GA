'''Bandwidth graph
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
    fitness = 1 - (N * Pdata) / (Bavail * T * 100)  # Example fitness function
    return fitness

# Initialize Population
population = initialize_population(population_size)
best_fitnesses = []

# Main Genetic Algorithm Loop
for generation in range(max_generations):
    # Evaluate Fitness of each chromosome in Population
    fitness_scores = [fitness_evaluation(chromosome) for chromosome in population]
    best_fitness = max(fitness_scores)
    best_fitnesses.append(best_fitness)

    # Other genetic algorithm steps...
    # For brevity, the rest of the genetic algorithm code is omitted.

# Store the best chromosome from the genetic algorithm
best_chromosome_idx = fitness_scores.index(max(fitness_scores))
best_chromosome = population[best_chromosome_idx]
ga_bandwidth_allocations = best_chromosome[:5]  # Assuming first 5 genes represent bandwidth allocations

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

# Use the trained neural network to predict bandwidth allocations
predicted_bandwidth_allocations = mlp.predict(X_test_scaled)

# Apply bandwidth allocation algorithms
pf_allocation = [(demand / sum(traffic_demands)) * total_bandwidth for demand in traffic_demands]
mm_allocation = [min(total_bandwidth, demand) for demand in traffic_demands]
wfq_allocation = [(weight / sum([1, 2, 3, 1, 2])) * total_bandwidth for weight in [1, 2, 3, 1, 2]]  # Example weights

# Define the category names
categories = ['Throughput', 'Latency', 'Packet Loss Rate', 'Fairness', 'QoS', 'Energy Consumption']

# Predicted results for each category
predicted_results = {
    'Throughput': predicted_bandwidth_allocations[:4],
    'Latency': predicted_bandwidth_allocations[4:8],
    'Packet Loss Rate': predicted_bandwidth_allocations[8:12],
    'Fairness': predicted_bandwidth_allocations[12:16],
    'QoS': predicted_bandwidth_allocations[16:20],
    'Energy Consumption': predicted_bandwidth_allocations[20:]
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
r1 = np.arange(len(traffic_demands))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]

# Create bars for Genetic Algorithm
plt.bar(r1, ga_bandwidth_allocations, color='purple', width=bar_width, edgecolor='grey', label='Genetic Algorithm')

# Create bars for Neural Network Predictions
plt.bar(r2, pf_allocation, color='b', width=bar_width, edgecolor='grey', label='Proportional Fairness')
plt.bar(r3, mm_allocation, color='g', width=bar_width, edgecolor='grey', label='Max-Min Fairness')
plt.bar(r4, wfq_allocation, color='r', width=bar_width, edgecolor='grey', label='Weighted Fair Queuing')

# Add xticks on the middle of the group bars
plt.xlabel('Users/Applications', fontweight='bold')
plt.ylabel('Bandwidth Allocation', fontweight='bold')
plt.xticks([r + bar_width for r in range(len(traffic_demands))], [f'User/App {i+1}' for i in range(len(traffic_demands))])

# Create legend & Show graphic
plt.legend()
plt.title('Bandwidth Allocation Comparison')
plt.tight_layout()
plt.show()'''
'''import numpy as np
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
    fitness = 1 - (N * Pdata) / (Bavail * T * 100)  # Example fitness function
    return fitness

# Initialize Population
population = initialize_population(population_size)
best_fitnesses = []

# Main Genetic Algorithm Loop
for generation in range(max_generations):
    # Evaluate Fitness of each chromosome in Population
    fitness_scores = [fitness_evaluation(chromosome) for chromosome in population]
    best_fitness = max(fitness_scores)
    best_fitnesses.append(best_fitness)

    # Other genetic algorithm steps...
    # For brevity, the rest of the genetic algorithm code is omitted.

# Store the best chromosome from the genetic algorithm
best_chromosome_idx = fitness_scores.index(max(fitness_scores))
best_chromosome = population[best_chromosome_idx]
ga_bandwidth_allocations = best_chromosome[:5]  # Assuming first 5 genes represent bandwidth allocations

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

# Use the trained neural network to predict energy consumption
predicted_energy_consumption = mlp.predict(X_test_scaled)[:len(traffic_demands)]  # Only take predictions for the number of users/applications

# Plotting the comparison graph for Energy Consumption
plt.figure(figsize=(12, 8))

# Bar width
bar_width = 0.35

# Position of bars on x-axis
r1 = np.arange(len(traffic_demands))

# Create bars for predicted energy consumption
plt.bar(r1, predicted_energy_consumption, color='purple', width=bar_width, edgecolor='grey', label='Predicted Energy Consumption')

# Add xticks on the middle of the group bars
plt.xlabel('Users/Applications', fontweight='bold')
plt.ylabel('Energy Consumption', fontweight='bold')
plt.xticks([r for r in range(len(traffic_demands))], [f'User/App {i+1}' for i in range(len(traffic_demands))])

# Create legend & Show graphic
plt.legend()
plt.title('Predicted Energy Consumption Comparison')
plt.tight_layout()
plt.show()'''
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
    fitness = 1 - (N * Pdata) / (Bavail * T * 100)  # Example fitness function
    return fitness

# Initialize Population
population = initialize_population(population_size)
best_fitnesses = []

# Main Genetic Algorithm Loop
for generation in range(max_generations):
    # Evaluate Fitness of each chromosome in Population
    fitness_scores = [fitness_evaluation(chromosome) for chromosome in population]
    best_fitness = max(fitness_scores)
    best_fitnesses.append(best_fitness)

    # Other genetic algorithm steps...
    # For brevity, the rest of the genetic algorithm code is omitted.

# Store the best chromosome from the genetic algorithm
best_chromosome_idx = fitness_scores.index(max(fitness_scores))
best_chromosome = population[best_chromosome_idx]
ga_bandwidth_allocations = best_chromosome[:5]  # Assuming first 5 genes represent bandwidth allocations

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

# Use the trained neural network to predict bandwidth allocations
predicted_bandwidth_allocations = mlp.predict(X_test_scaled)

# Apply bandwidth allocation algorithms
pf_allocation = [(demand / sum(traffic_demands)) * total_bandwidth for demand in traffic_demands]
mm_allocation = [min(total_bandwidth, demand) for demand in traffic_demands]
wfq_allocation = [(weight / sum([1, 2, 3, 1, 2])) * total_bandwidth for weight in [1, 2, 3, 1, 2]]  # Example weights

# Define the category names
categories = ['Throughput', 'Latency', 'Packet Loss Rate', 'Fairness', 'QoS', 'Energy Consumption']

# Predicted results for QoS category
predicted_qos_allocations = [(allocation / sum(predicted_bandwidth_allocations[16:20])) * total_bandwidth for allocation in predicted_bandwidth_allocations[16:20]]

# Plotting the comparison graph for QoS category
plt.figure(figsize=(8, 6))

# Bar width
bar_width = 0.2

# Position of bars on x-axis
r1 = np.arange(len(traffic_demands))

# Create bars for Genetic Algorithm and Neural Network Predictions
plt.bar(r1, ga_bandwidth_allocations, color='purple', width=bar_width, edgecolor='grey', label='Genetic Algorithm')
plt.bar(r1 + bar_width, predicted_qos_allocations, color='b', width=bar_width, edgecolor='grey', label='Neural Network Predictions')

# Add xticks on the middle of the group bars
plt.xlabel('Users/Applications', fontweight='bold')
plt.ylabel('QoS Bandwidth Allocation', fontweight='bold')
plt.xticks([r + bar_width/2 for r in range(len(traffic_demands))], [f'User/App {i+1}' for i in range(len(traffic_demands))])

# Create legend & Show graphic
plt.legend()
plt.title('QoS Bandwidth Allocation Comparison')
plt.tight_layout()
plt.show()
