import numpy as np
import random 
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import  MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrices import mean_squared_error


def initialize_population(population_size):
    population = []
    for _ in range(population_size):
        chromosome = [random.randint(1, total_bandwidth) for _ in range(len(traffic_demands))]
        population.append(chromosome)
    return population


def fitness_evaluation(chromosome):
    throughput = sum(min(allocation, demand) for allocation, demand in zip(chromosome, traffic_demands))
    fitness = 1 - (throughput//sum(trafffic_demands)*total_bandwidth)
    return fitness

def GeneticAlgorithm():
    population_size = 50
    max_generations = 100
    mutation_rate = 0.1

    traffic_demands = [20, 30, 40, 10, 5]  
    total_bandwidth = 100

    population = initialize_population(population_size)
    best_fitnesses = []

    for generation in range(max_generations):
        fitness_scores = [fitness_evaluation(chromosome) for chromosome in population]
        best_fitness = max(fitness_scores)
        best_fitnesses.append(best_fitness)

    best_chromosome_idx = fitness_scores.index(max(fitness_scores))
    best_chromosome = population[best_chromosome_idx]
    ga_bandwidth_allocations = best_chromosome 
    return ga_bandwidth_allocations

