import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
import filterpy.stats as stats
from collections import namedtuple


# Plot the Gaussian PDF
p = stats.plot_gaussian_pdf(mean=10., variance=1., xlim=(4, 16), ylim=(0, .5))

# Show the plot
plt.show()

xs = range(500)
ys = randn(500)*1. + 10.
plt.plot(xs, ys)
print(f'Mean of readings is {np.mean(ys):.3f}')
plt.show()

gaussian = namedtuple('Gaussian', ['mean', 'var'])
gaussian.__repr__ = lambda s: f'𝒩(μ={s[0]:.3f}, 𝜎²={s[1]:.3f})'


def predict(pos, movement):
    return gaussian(pos.mean + movement.mean, pos.var + movement.var)


pos = gaussian(10., .2**2)
move = gaussian(15., .7**2)
pr = predict(pos, move)
print(pr)