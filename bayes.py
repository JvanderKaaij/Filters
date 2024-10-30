import numpy as np
import matplotlib.pyplot as plt


def calculate_likelihood(env, z, z_prob):
    """ compute likelihood that a measurement matches
    positions in the environment.

    Parameters:
        - env: (array) a map of the environment
        - z: (float) the current measurement
        - z_prob: (float) the probability of the measurement being correct
    """

    try:
        scale = z_prob / (1. - z_prob)
    except ZeroDivisionError:
        scale = 1e8

    likelihood = np.ones(len(env))
    likelihood[env == z] *= scale
    return likelihood


# Define discretely where are doors and where are wall in the hallway
hallway = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 0])
prior = np.array([.1]*10)
kernel = (.1, .8, .1)


def predict(pdf, offset, kernel):
    N = len(pdf)
    kN = len(kernel)
    width = int((kN - 1) / 2)

    prior = np.zeros(N)
    for i in range(N):
        for k in range (kN):
            index = (i + (width-k) - offset) % N
            prior[i] += pdf[index] * kernel[k]
    return prior


def normalize(array):
    total = np.sum(array)
    if total == 0:
        return array
    return array / total


def update(likelihood, _prior):
    """Normalize the likelihood so that it sums to 1 and multiply it by the prior."""
    return normalize(likelihood * _prior)


def main():
    plt.bar(range(len(prior)), prior, tick_label=np.round(prior, 1))
    plt.title("Prior")
    plt.ylim(0, .5)
    plt.ylabel("Frequency")
    plt.show()

    likelihood = calculate_likelihood(hallway, z=1, z_prob=.75)
    posterior = update(likelihood, prior)

    plt.bar(range(len(posterior)), posterior, tick_label=np.round(posterior, 2))
    plt.title("Posterior")
    plt.ylim(0, .5)
    plt.ylabel("Frequency")
    plt.show()

    # Predict the next state

    prior2 = predict(posterior, 1, kernel)

    plt.bar(range(len(prior2)), prior2, tick_label=np.round(prior2, 2))
    plt.title("Prior 2")
    plt.ylim(0, .5)
    plt.ylabel("Frequency")
    plt.show()

    likelihood = calculate_likelihood(hallway, z=1, z_prob=.75)
    posterior2 = update(likelihood, prior2)

    plt.bar(range(len(posterior2)), posterior2, tick_label=np.round(posterior2, 2))
    plt.title("Posterior 2")
    plt.ylim(0, .5)
    plt.ylabel("Frequency")
    plt.show()

    # Predict the next state

    prior3 = predict(posterior2, 1, kernel)

    plt.bar(range(len(prior3)), prior3, tick_label=np.round(prior3, 2))
    plt.title("Prior 3")
    plt.ylim(0, .5)
    plt.ylabel("Frequency")
    plt.show()

    likelihood = calculate_likelihood(hallway, z=0, z_prob=.75)  # NOTICE Here that we now measure a 0 as we are along a wall!
    posterior3 = update(likelihood, prior3)

    plt.bar(range(len(posterior3)), posterior3, tick_label=np.round(posterior3, 2))
    plt.title("Posterior 3")
    plt.ylim(0, .5)
    plt.ylabel("Frequency")
    plt.show()


if __name__ == '__main__':
    main()
