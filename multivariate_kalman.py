import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter


np.random.seed(52)
delta_time = 1.0
time = np.arange(0, 400, delta_time)


def generate_noisy_data(start, growth_rate, noise_std, noise_mean=0.0):
    values = growth_rate * time + start
    noise = np.random.normal(noise_mean, noise_std, size=time.shape)
    noise_values = values + noise
    return values, noise_values


def filter_data_single(y_noisy):
    kf = KalmanFilter(dim_x=1, dim_z=1)  # 1 state variable x, 1 measurement z
    kf.x = np.array([0.0])  # initial state
    kf.F = np.array([[1.0]])  # state transition matrix
    kf.H = np.array([[1.0]])  # Measurement function
    kf.Q = np.array([[0.005]])  # Assumed process noise (adjust as needed)
    kf.R = np.array([[0.4]])  # Assumed measurement noise (adjust as needed)
    kf.P = np.array([[1]])  # Initial uncertainty in the state estimate
    predictions = []
    for z in y_noisy:
        kf.predict()
        kf.update(z)
        predictions.append(kf.x[0])
    return predictions


def filter_data_multi(y_noisy):
    kf = KalmanFilter(dim_x=2, dim_z=1)  # 2 state variables position and velocity, 1 measurement z
    kf.x = np.array([0.0, 0.0])  # initial state
    kf.F = np.array([[1.0, delta_time],
                     [0.0, 1.0]])  # state transition matrix
    kf.H = np.array([[1.0, 0.0]])  # Measurement function
    kf.Q = np.array([[0.0001, 0.0],
                     [0.0, 0.001]])  # Assumed process noise (adjust as needed)
    kf.R = np.array([[0.4]])  # Assumed measurement noise (adjust as needed)
    kf.P = np.array([[1, 0],
                     [0, 1]])  # Initial uncertainty in the state estimate
    predictions = []
    velocities = []
    for z in y_noisy:
        kf.predict()
        kf.update(z)
        predictions.append(kf.x[0])
        velocities.append(kf.x[1])
    return predictions, velocities


def main():
    y, y_noisy = generate_noisy_data(10, 3, 90)
    y_filtered_single = filter_data_single(y_noisy)

    y_filtered_multi_pred, y_filtered_multi_vel = filter_data_multi(y_noisy)

    plt.scatter(time, y_noisy, label="Noisy Data", alpha=0.2)
    plt.plot(time, y, label="True Line (y=3x+7)", color="red", linewidth=2)
    plt.plot(time, y_filtered_multi_pred, label="Filtered Position", alpha=0.7, color="green")
    plt.plot(time, y_filtered_multi_vel, label="Filtered Velocity", alpha=0.7, color="pink")
    plt.xlabel("Time")
    plt.ylabel("Y")
    plt.title("Linear Data with Noise")
    plt.legend()
    plt.show()
    pass


if __name__ == '__main__':
    main()