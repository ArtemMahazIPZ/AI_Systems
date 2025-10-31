import numpy as np
import matplotlib.pyplot as plt

x = np.array([0, 5, 10, 15, 20, 25], dtype=float)
y = np.array([21, 39, 51, 63, 70, 90], dtype=float)

X = np.column_stack([np.ones_like(x), x])
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
b0, b1 = beta
y_hat = b0 + b1*x
r2 = 1 - np.sum((y - y_hat)**2)/np.sum((y - y.mean())**2)
mse = np.mean((y - y_hat)**2)

print(f"Рівняння регресії: y = {b0:.6f} + {b1:.6f}x")
print(f"R² = {r2:.6f},  MSE = {mse:.6f}")

xx = np.linspace(x.min(), x.max(), 400)
yy = b0 + b1*xx
plt.scatter(x, y, label="Експериментальні дані")
plt.plot(xx, yy, label="Лінія МНК")
plt.title("Завдання 2. Лінійна регресія (Варіант 1)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.show()


