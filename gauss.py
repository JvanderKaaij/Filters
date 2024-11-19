import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy.random import randn

# Define parameters for the Gaussian
mean = 2
std_dev = 3

# Generate x values (range around the mean)
x = np.linspace(mean - 4 * std_dev, mean + 4 * std_dev, 1000)

# Calculate the PDF for each x value
pdf_values = norm(mean, std_dev).pdf(x)

# Plot the Gaussian curve
plt.plot(x, pdf_values, label=f'Gaussian PDF (mean={mean}, std dev={std_dev})')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title('Gaussian Distribution')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()


def sense():
    return 10 + randn()*2


zs = [sense() for i in range(5000)]
plt.plot(zs, lw=1)
plt.show()