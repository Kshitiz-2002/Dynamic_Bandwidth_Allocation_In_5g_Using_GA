import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
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

# Fitness Evaluation (Original GA)
def fitness_evaluation(chromosome):
    Bavail = chromosome[5]  # Free bandwidth available in the cell or neighboring cell
    Bfair = chromosome[4]  # Fair bandwidth allocated for real-time users
    T = chromosome[3]  # Time slot by the real-time user
    N = chromosome[0]  # Number of users generating calls with a particular service
    Pdata = chromosome[2]  # Data packets of real-time users
    latency = (N * Pdata) / (Bavail * T * 100)  # Example latency calculation
    return latency

# Main Genetic Algorithm Loop
population = initialize_population(population_size)
best_latencies_ga = []
best_latencies_ga_rnn = []

for generation in range(max_generations):
    # Evaluate Fitness of each chromosome in Population for both GA and GA+RNN
    fitness_scores_ga = [fitness_evaluation(chromosome) for chromosome in population]
    best_latency_ga = min(fitness_scores_ga)
    best_latencies_ga.append(best_latency_ga)

    # Other genetic algorithm steps...
    # For brevity, the rest of the genetic algorithm code is omitted.

    # Genetic Algorithm with RNN
    # Here, you would integrate RNN predictions into fitness evaluation for each chromosome

    # Store best latency for GA with RNN
    best_latency_ga_rnn = best_latency_ga  # Example, replace with actual implementation
    best_latencies_ga_rnn.append(best_latency_ga_rnn)

# Generate latency allocations for other algorithms (for demonstration)
pf_allocation = [(demand / sum(traffic_demands)) * total_bandwidth for demand in traffic_demands]
mm_allocation = [min(total_bandwidth, demand) for demand in traffic_demands]
wfq_allocation = [(weight / sum([1, 2, 3, 1, 2])) * total_bandwidth for weight in [1, 2, 3, 1, 2]]

# Plotting the comparison graph
generations = range(1, max_generations + 1)

plt.figure(figsize=(16, 6))

# Plot for Genetic Algorithm (GA)
plt.subplot(1, 2, 1)
plt.plot(generations, best_latencies_ga, label='Genetic Algorithm (GA)', color='blue')
plt.xlabel('Generation')
plt.ylabel('Best Latency (ms)')
plt.title('Performance Comparison: Genetic Algorithm (GA)')
plt.grid(True)
plt.legend()

# Plot for Genetic Algorithm with RNN (GA+RNN)
plt.subplot(1, 2, 2)
plt.plot(generations, best_latencies_ga_rnn, label='Genetic Algorithm with RNN (GA+RNN)', color='orange')
plt.xlabel('Generation')
plt.ylabel('Best Latency (ms)')
plt.title('Performance Comparison: Genetic Algorithm with RNN (GA+RNN)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
