import numpy as np
import time

class Genetic:
    def __init__(self, pop_size, n_genes, num_generations, num_parents_mating, usage):
        self.pop_size = pop_size
        self.n_genes = n_genes
        self.num_generations = num_generations
        self.num_parents_mating = num_parents_mating
        self.usage = usage
        self.population = self.generate_population()

    def fitness_function(self, bw_allocation, user_usage):
        return -np.sum((user_usage - bw_allocation) ** 2)

    def generate_population(self):
        return np.random.rand(self.pop_size, self.n_genes)

    def select(self, fitness):
        parents = np.empty((self.num_parents_mating, self.population.shape[1]))
        for parent_num in range(self.num_parents_mating):
            max_fitness_idx = np.where(fitness == np.max(fitness))
            max_fitness_idx = max_fitness_idx[0][0]
            parents[parent_num, :] = self.population[max_fitness_idx, :]
            fitness[max_fitness_idx] = -99999999999
        return parents

    def crossover(self, parents):
        offspring_size = (self.pop_size - parents.shape[0], self.n_genes)
        offspring = np.empty(offspring_size)
        crossover_point = np.uint8(offspring_size[1] / 2)

        for k in range(offspring_size[0]):
            parent1_idx = k % parents.shape[0]
            parent2_idx = (k + 1) % parents.shape[0]
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]

        return offspring

    def mutate(self, offspring_crossover, num_mutations=1):
        for idx in range(offspring_crossover.shape[0]):
            for _ in range(num_mutations):
                mutation_idx = np.random.randint(offspring_crossover.shape[1])
                offspring_crossover[idx, mutation_idx] = np.random.rand()
        return offspring_crossover

    def run(self):
        start_time = time.time()
        for generation in range(self.num_generations):
            fitness = np.array([self.fitness_function(ind, self.usage[:, generation % self.usage.shape[1]]) for ind in self.population])
            parents = self.select(fitness)
            offspring_crossover = self.crossover(parents)
            offspring_mutation = self.mutate(offspring_crossover, num_mutations=2)
            self.population[0:parents.shape[0], :] = parents
            self.population[parents.shape[0]:, :] = offspring_mutation
        end_time = time.time()

        best_solution = self.population[np.argmax([self.fitness_function(ind, self.usage[:, generation % self.usage.shape[1]]) for ind in self.population])]

        return best_solution, end_time - start_time