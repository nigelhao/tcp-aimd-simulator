import numpy as np
import matplotlib.pyplot as plt

NUM_OF_FLOW = 100
CWND_SIZE = 10000

ALPHA = 1
BETA = 0

SS_THRESH = 0.6
SS_ALPHA = 2

ITER_LIMIT = CWND_SIZE * 2


def linear_AI(flow_allocations):
    flow_allocations += ALPHA
    return flow_allocations


def linear_MD(flow_allocations):
    flow_allocations *= BETA
    return flow_allocations


def nonlinear_SS(flow_allocations, ss_n):
    flow_allocations += SS_ALPHA**ss_n
    return flow_allocations


def graph_flow_allocation(allocation_history):

    x = y = np.linspace(0, CWND_SIZE, 400)
    w = np.linspace(0, CWND_SIZE, 400)
    v = -w + CWND_SIZE
    plt.figure(2)
    plt.plot(
        allocation_history[:, 0], allocation_history[:, 1], label="Flow Allocations"
    )
    plt.plot(x, y, label="Fairness line", linestyle="dashed")
    plt.plot(v, w, label="Efficiency line", linestyle="dashed")
    plt.xlabel("Flow 1")
    plt.ylabel("Flow 2")
    plt.title("Flow Allocations")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, CWND_SIZE)
    plt.ylim(0, CWND_SIZE)
    plt.show()


def graph_cwnd_iternation(allocation_history):
    plt.figure(1)
    for i in range(NUM_OF_FLOW):
        plt.plot(allocation_history[:, i])

    plt.xlabel("Iteration")
    plt.ylabel("CWND")
    plt.title("CWND per flow over Iteration")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, ITER_LIMIT)
    plt.ylim(0, 3 * (CWND_SIZE / NUM_OF_FLOW))
    plt.show()


def calculate_valid_windows(allocation_history):
    total = 0
    for flows in allocation_history:
        i_total = np.sum(flows)
        total += round(i_total) if i_total < CWND_SIZE else CWND_SIZE

    return total


def simulate(flow_allocations):

    allocation_history = []
    ss_n = 0
    for _ in range(ITER_LIMIT):

        allocation_history.append(flow_allocations.copy())
        if np.sum(flow_allocations) <= (CWND_SIZE * SS_THRESH):
            ss_n += 1
            flow_allocations = nonlinear_SS(flow_allocations, ss_n)
        elif np.sum(flow_allocations) <= CWND_SIZE:
            ss_n = 0
            flow_allocations = linear_AI(flow_allocations)
        else:
            ss_n = 0
            flow_allocations = linear_MD(flow_allocations)

    return np.array(allocation_history)


def main():
    np.set_printoptions(precision=10)
    flow_allocations = np.random.randint(0, CWND_SIZE, size=NUM_OF_FLOW).astype(float)

    flow_allocations[0] = 1050
    flow_allocations[1] = 5500

    allocation_history = simulate(flow_allocations)

    print(calculate_valid_windows(allocation_history))

    if NUM_OF_FLOW == 2:
        graph_flow_allocation(allocation_history)

    graph_cwnd_iternation(allocation_history)


main()
