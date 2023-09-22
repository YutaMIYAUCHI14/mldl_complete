import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(start = -10, stop = 10, num = 2_000).reshape(-1, 1)
y = np.squeeze(20*x*np.exp(-np.abs(x)))

rng = np.random.RandomState(1)
training_indices = rng.choice(np.arange(y.size), size = 50, replace = False)
X_train, y_train = x[training_indices], y[training_indices]

noise_std = 1.0
y_train_obs = y_train + rng.normal(loc = 0.0, scale = noise_std, size = y_train.shape)

plt.figure(0)
plt.title("True Generative and Observed Signals")
plt.grid(True)
plt.plot(x, y, label = "True Signal")
plt.scatter(X_train, y_train_obs, label = "Observed Signal")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

RBF_kernel = 1*RBF(length_scale = 1.0, length_scale_bounds = (1e-2, 1e2))
gaussian_process = GaussianProcessRegressor(
    kernel = RBF_kernel,
    alpha = noise_std**2,
    n_restarts_optimizer = 9,
)
gaussian_process.fit(X_train, y_train_obs)

print(gaussian_process.kernel_)

mean_prediction, std_prediction = gaussian_process.predict(x, return_std = True)

plt.figure(1)
plt.title("Prediction Result")
plt.grid(True)
plt.plot(x, y, label = "True Signal")
plt.errorbar(
    X_train,
    y_train_obs,
    noise_std,
    linestyle = "None",
    color = "tab:blue",
    marker = ".",
    markersize = 10,
    label = "Observation",
)
plt.plot(x, mean_prediction, label = "Mean Prediction", linestyle = "dotted")
plt.fill_between(
    x.ravel(),
    mean_prediction - 1.96*std_prediction,
    mean_prediction + 1.96*std_prediction,
    color = "tab:orange",
    alpha = 0.5,
    label = r"95% cofidence interval",
)
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()