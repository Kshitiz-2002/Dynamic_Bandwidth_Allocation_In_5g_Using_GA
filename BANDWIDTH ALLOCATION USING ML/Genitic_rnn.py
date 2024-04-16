'''import numpy as np
import random
import matplotlib.pyplot as plt

# Assuming you have the RNN implemented separately
# You can replace this with your RNN implementation
class RNNBandwidthPredictor:
    def __init__(self):
        # Assuming you have initialized your RNN model
        pass

    def predict(self, X):
        # Assuming X is input data, you should implement your prediction logic here
        # For simplicity, let's return random predictions
        return np.random.rand(X.shape[0], 5)

# Genetic Algorithm Parameters
population_size = 50
max_generations = 100
mutation_rate = 0.1

# Initialize Population
def initialize_population(population_size):
    population = []
    for _ in range(population_size):
        chromosome = [random.randint(1, 50) for _ in range(8)]  # Random values for each gene
        population.append(chromosome)
    return population

# Fitness Evaluation
def fitness_evaluation(chromosome):
    # Example fitness function: sum of all values in the chromosome
    fitness = sum(chromosome)
    return fitness

# Selection
def selection(population, fitness_scores):
    sorted_population = [x for _, x in sorted(zip(fitness_scores, population), reverse=True)]
    return sorted_population[:len(sorted_population) // 2]

# Crossover
def crossover(parent1, parent2):
    crossover_point = random.randint(0, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Mutation
def mutation(chromosome):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = random.randint(1, 50)
    return chromosome

# Initialize Population
population = initialize_population(population_size)
best_fitnesses_genetic = []
best_fitnesses_rnn_genetic = []

# Create RNN predictor
rnn_predictor = RNNBandwidthPredictor()

# Main Genetic Algorithm Loop
for generation in range(max_generations):
    # Evaluate Fitness of each chromosome in Population for Genetic only
    fitness_scores_genetic = [fitness_evaluation(chromosome) for chromosome in population]
    best_fitness_genetic = max(fitness_scores_genetic)
    best_fitnesses_genetic.append(best_fitness_genetic)

    # Evaluate Fitness of each chromosome in Population for RNN + Genetic
    fitness_scores_rnn_genetic = [fitness_evaluation(chromosome) for chromosome in population]
    best_fitness_rnn_genetic = max(fitness_scores_rnn_genetic)
    best_fitnesses_rnn_genetic.append(best_fitness_rnn_genetic)

    # Selection
    selected_population = selection(population, fitness_scores_genetic)
    
    # Crossover
    offspring_population = []
    for _ in range(population_size - len(selected_population)):
        parent1, parent2 = random.sample(selected_population, 2)
        child1, child2 = crossover(parent1, parent2)
        offspring_population.extend([child1, child2])
    
    # Mutation
    population = [mutation(chromosome) for chromosome in offspring_population]

# Plotting the comparison graph
plt.figure(figsize=(12, 8))

# Plot Genetic Algorithm
plt.plot(range(max_generations), best_fitnesses_genetic, label='Genetic Algorithm')

# Plot Genetic Algorithm with RNN
plt.plot(range(max_generations), best_fitnesses_rnn_genetic, label='Genetic Algorithm with RNN')

# Add labels and title
plt.xlabel('Generation', fontweight='bold')
plt.ylabel('Best Fitness Score', fontweight='bold')
plt.title('Genetic Algorithm vs Genetic Algorithm with RNN')

# Add legend
plt.legend()

# Show plot
plt.grid(True)
plt.show()'''
'''import numpy as np
import random
import matplotlib.pyplot as plt

# Genetic Algorithm Parameters
population_size = 50
max_generations = 100
mutation_rate = 0.1

# Initialize Population
def initialize_population(population_size):
    population = []
    for _ in range(population_size):
        chromosome = [random.randint(1, 50) for _ in range(8)]  # Random values for each gene
        population.append(chromosome)
    return population

# Fitness Evaluation
def fitness_evaluation(chromosome):
    # Example fitness function: sum of all values in the chromosome
    fitness = sum(chromosome)
    return fitness

# Selection
def selection(population, fitness_scores):
    sorted_population = [x for _, x in sorted(zip(fitness_scores, population), reverse=True)]
    return sorted_population[:len(sorted_population) // 2]

# Crossover
def crossover(parent1, parent2):
    crossover_point = random.randint(0, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Mutation
def mutation(chromosome):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = random.randint(1, 50)
    return chromosome

# Initialize Population
population = initialize_population(population_size)
best_fitnesses_genetic = []
best_fitnesses_rnn_genetic = []

# Main Genetic Algorithm Loop
for generation in range(max_generations):
    # Evaluate Fitness of each chromosome in Population
    fitness_scores = [fitness_evaluation(chromosome) for chromosome in population]  # For Genetic only
    
    best_fitness_genetic = max(fitness_scores)
    best_fitnesses_genetic.append(best_fitness_genetic)

    # Selection
    selected_population = selection(population, fitness_scores)
    
    # Crossover
    offspring_population = []
    for _ in range(population_size - len(selected_population)):
        parent1, parent2 = random.sample(selected_population, 2)
        child1, child2 = crossover(parent1, parent2)
        offspring_population.extend([child1, child2])
    
    # Mutation
    population = [mutation(chromosome) for chromosome in offspring_population]

# Plotting the comparison graph
plt.figure(figsize=(12, 8))

# Plot Genetic Algorithm
plt.plot(range(max_generations), best_fitnesses_genetic, label='Genetic Algorithm')

# Add labels and title
plt.xlabel('Generation', fontweight='bold')
plt.ylabel('Best Fitness Score', fontweight='bold')
plt.title('Genetic Algorithm')

# Add legend
plt.legend()

# Set y-axis limits to ensure visibility
plt.ylim(min(best_fitnesses_genetic), max(best_fitnesses_genetic) + 10)

# Show plot
plt.grid(True)
plt.show()'''
import numpy as np
import random
import matplotlib.pyplot as plt

# Genetic Algorithm Parameters
population_size = 50
max_generations = 100
mutation_rate = 0.1

# Initialize Population
def initialize_population(population_size):
    population = []
    for _ in range(population_size):
        chromosome = [random.randint(1, 50) for _ in range(8)]  # Random values for each gene
        population.append(chromosome)
    return population

# Fitness Evaluation for Genetic Algorithm
def fitness_evaluation_genetic(chromosome):
    # Example fitness function: sum of all values in the chromosome
    fitness = sum(chromosome)
    return fitness

# Fitness Evaluation for Genetic Algorithm with RNN
def fitness_evaluation_rnn_genetic(chromosome):
    # Here you would evaluate the fitness using the RNN
    # Example: fitness = rnn.evaluate(chromosome)
    # Replace this with your RNN evaluation code
    fitness = sum(chromosome) * 2  # Placeholder example
    return fitness

# Selection
def selection(population, fitness_scores):
    sorted_population = [x for _, x in sorted(zip(fitness_scores, population), reverse=True)]
    return sorted_population[:len(sorted_population) // 2]

# Crossover
def crossover(parent1, parent2):
    crossover_point = random.randint(0, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Mutation
def mutation(chromosome):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = random.randint(1, 50)
    return chromosome

# Initialize Population
population = initialize_population(population_size)
best_fitnesses_genetic = []
best_fitnesses_rnn_genetic = []

# Main Genetic Algorithm Loop
for generation in range(max_generations):
    # Evaluate Fitness of each chromosome in Population
    fitness_scores_genetic = [fitness_evaluation_genetic(chromosome) for chromosome in population]
    fitness_scores_rnn_genetic = [fitness_evaluation_rnn_genetic(chromosome) for chromosome in population]
    
    best_fitness_genetic = max(fitness_scores_genetic)
    best_fitness_rnn_genetic = max(fitness_scores_rnn_genetic)
    best_fitnesses_genetic.append(best_fitness_genetic)
    best_fitnesses_rnn_genetic.append(best_fitness_rnn_genetic)

    # Selection
    selected_population = selection(population, fitness_scores_genetic)
    
    # Crossover
    offspring_population = []
    for _ in range(population_size - len(selected_population)):
        parent1, parent2 = random.sample(selected_population, 2)
        child1, child2 = crossover(parent1, parent2)
        offspring_population.extend([child1, child2])
    
    # Mutation
    population = [mutation(chromosome) for chromosome in offspring_population]

# Plotting the results
generations = range(max_generations)
plt.plot(generations, best_fitnesses_genetic, label='GMBA')
plt.plot(generations, best_fitnesses_rnn_genetic, label='GMBA with RNN')
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.title('Comparison of Genetic Algorithms')
plt.legend()
plt.show()
