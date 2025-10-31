import numpy as np
import matplotlib.pyplot as plt

x = np.array([0.1, 0.3, 0.4, 0.6, 0.7])
y = np.array([3.2, 3.0, 1.0, 1.8, 1.9])

V = np.vander(x, N=5, increasing=False)
coeffs, *_ = np.linalg.lstsq(V, y, rcond=None)

def p(xx): return np.polyval(coeffs, xx)

y02, y05 = p(0.2), p(0.5)
print("Коефіцієнти полінома:", np.round(coeffs, 6))
print(f"f(0.2) = {y02:.6f}")
print(f"f(0.5) = {y05:.6f}")

xx = np.linspace(0.05, 0.75, 400)
plt.scatter(x, y, label="Вихідні точки")
plt.plot(xx, p(xx), label="Поліном 4-го степеня")
plt.title("Завдання 3. Інтерполяція поліномом 4-го степеня")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.show()


