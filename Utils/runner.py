import torch
from Algorithms.Access_Aware import AccessAware
from Algorithms.Genetic import Genetic
from Utils.train import  train_model
import numpy as np
import matplotlib.pyplot as plt

def run_access_aware_algorithm():
    model, x_test, y_test, criterion = train_model()

    model.eval()
    with torch.no_grad():
        test_outputs = model(x_test)
        test_loss = criterion(test_outputs, y_test)
        print(f'Test Loss: {test_loss.item():.4f}')

    # Generate synthetic data for Access-Aware input
    numMBS = 3
    numMC = 100
    nMC = [10, 15, 20]
    B = [100, 200, 300]

    # Use the trained NARNET model to predict the required bandwidth
    R = []
    for i in range(numMBS):
        with torch.no_grad():
            predictions = model(x_test).numpy().flatten().tolist()
        R.append(predictions[:nMC[i]])  # Ensure each list in R is the correct length

    bm = 1.0
    bM = 10.0

    access_aware = AccessAware(numMBS, numMC, nMC, B, R, bm, bM)
    S = 10.0
    N = 5.0
    Bi, bij = access_aware.dynamic_allocation(S, N)

    print(f'Allocated Bandwidths: {Bi}')
    print(f'Required Bandwidths: {R}')
    print(f'Bandwidth Allocation Matrix: {bij}')


def run_genetic_algorithm():
    traffic = np.load('../traffic.npy')
    usage = np.load('../usage.npy')

    pop_size = 50
    n_genes = usage.shape[0]
    num_generations = 100
    num_parents_mating = 25

    ga = Genetic(pop_size, n_genes, num_generations, num_parents_mating, usage)
    best_solution, training_time = ga.run()

    time_points = len(traffic)
    utilization = []
    unsatisfied_cells = []
    throughput = []
    latency = []
    packet_loss = []

    for t in range(time_points):
        current_usage = usage[:, t]
        predicted_bw = best_solution * traffic[t]
        utilization.append(np.sum(current_usage) / np.sum(predicted_bw))
        unsatisfied_cells.append(np.sum(current_usage > predicted_bw))
        throughput.append(np.sum(current_usage))
        latency.append(np.mean(current_usage / predicted_bw))
        packet_loss.append(np.sum(current_usage > predicted_bw) / n_genes)

    plt.figure(figsize=(12, 8))

    plt.subplot(321)
    plt.plot(utilization)
    plt.title('Utilization Factor')

    plt.subplot(322)
    plt.plot(unsatisfied_cells)
    plt.title('Number of Unsatisfied Cells')

    plt.subplot(323)
    plt.plot(throughput)
    plt.title('Throughput')

    plt.subplot(324)
    plt.plot(latency)
    plt.title('Latency')

    plt.subplot(325)
    plt.plot(packet_loss)
    plt.title('Packet Loss')

    plt.tight_layout()
    plt.savefig('ga_metrics.png')
    plt.show()

    print(f"GA Training Time: {training_time} seconds")
