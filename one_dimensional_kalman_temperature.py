from collections import namedtuple
from numpy.random import randn
import matplotlib.pyplot as plt


process_var = .05**2
voltage_std = 0.13
actual_voltage = 16.3

gaussian = namedtuple('Gaussian', ['mean', 'var'])
gaussian.__repr__ = lambda s: f'𝒩(μ={s[0]:.3f}, 𝜎²={s[1]:.3f})'


def update(prior, measurement):
    x, P = prior  # mean and variance of prior
    z, R = measurement  # mean and variance of measurement

    y = z - x  # residual
    K = P / (P + R)  # Kalman gain

    x = x + K * y  # posterior
    P = (1 - K) * P  # posterior variance
    return gaussian(x, P)


def predict(posterior, movement):
    x, P = posterior  # mean and variance of posterior
    dx, Q = movement  # mean and variance of movement
    x = x + dx
    P = P + Q
    return gaussian(x, P)


def volt(voltage, std):
    return voltage + (randn() * std)


x = gaussian(25., 1000.)
process_model = gaussian(0., process_var)
N = 50
zs = [volt(actual_voltage, voltage_std) for i in range(N)]
ps = []

estimates = []

for z in zs:
    prior = predict(x, process_model)
    x = update(prior, gaussian(z, voltage_std**2))
    estimates.append((x.mean))
    ps.append(x.var)


plt.plot(range(len(zs)), zs)
plt.show()

# Plot the Gaussian PDF
plt.plot(range(len(estimates)), estimates)
plt.show()

plt.plot(range(len(ps)), ps)
plt.show()
