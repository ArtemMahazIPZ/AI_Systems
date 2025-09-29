import numpy as np
import matplotlib.pyplot as plt

def step(z: float) -> int:
    return 1 if z > 0 else 0

def perceptron_xor(x1: int, x2: int):
    h1 = step(x1 - x2 - 0.5)
    h2 = step(x2 - x1 - 0.5)
    y_out = step(h1 + h2 - 0.5)
    return h1, h2, y_out

POINTS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

xs = np.linspace(0, 1, 201)
ys = np.linspace(0, 1, 201)
XX, YY = np.meshgrid(xs, ys)
Xb = (XX > 0.5).astype(int)
Yb = (YY > 0.5).astype(int)
xor_grid = (np.logical_or(Xb, Yb) & ~np.logical_and(Xb, Yb)).astype(int)

plt.figure(figsize=(7.5, 7.5), dpi=140)
plt.title("XOR через OR/AND: карта рішень", fontsize=16)
pcm = plt.pcolormesh(XX, YY, xor_grid, shading="nearest", cmap="coolwarm", alpha=0.85)
plt.scatter(POINTS[:, 0], POINTS[:, 1], c="k", s=90, zorder=3)
for x0, y0 in POINTS:
    val = int((x0 or y0) and not (x0 and y0))
    plt.text(x0 + 0.03, y0 + 0.03, f"y={val}", fontsize=12, weight="bold")
plt.xlabel("x1", fontsize=14, labelpad=6)
plt.ylabel("x2", fontsize=14, labelpad=6)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.gca().set_aspect("equal", adjustable="box")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

line_x = np.linspace(-0.1, 1.1, 200)
plt.figure(figsize=(9.5, 9.5), dpi=160)  # збільшене зображення
plt.title("Двошаровий перцептрон XOR: вхідний простір", fontsize=18, pad=10)
plt.plot(line_x, line_x - 0.5, label="x2 = x1 - 0.5 (h1)", linewidth=2)
plt.plot(line_x, line_x + 0.5, label="x2 = x1 + 0.5 (h2)", linewidth=2)

for x0, y0 in POINTS:
    h1_i, h2_i, y_i = perceptron_xor(int(x0), int(y0))
    plt.scatter([x0], [y0], c="k", s=110, zorder=3)
    plt.text(
        x0 + 0.035, y0 + 0.035,
        f"h1={h1_i}, h2={h2_i}, y={y_i}",
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.85)
    )

plt.xlabel("x1", fontsize=16, labelpad=8)
plt.ylabel("x2", fontsize=16, labelpad=8)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.legend(fontsize=14, loc="best")
plt.gca().set_aspect("equal", adjustable="box")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

hidden = np.array([perceptron_xor(int(x0), int(y0)) for x0, y0 in POINTS])
H = hidden[:, :2]
Y = hidden[:, 2]

plt.figure(figsize=(7.5, 7.5), dpi=140)
plt.title("Прихований простір і фінальна розділяюча пряма", fontsize=16)
colors = ["b" if yi == 0 else "r" for yi in Y]
plt.scatter(H[:, 0], H[:, 1], c=colors, s=120, edgecolors="k", zorder=3)
for (h1_i, h2_i), y_i in zip(H, Y):
    plt.text(
        h1_i + 0.03, h2_i + 0.03, f"y={y_i}",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85)
    )

hxs = np.linspace(-0.1, 1.1, 200)
plt.plot(hxs, 0.5 - hxs, label="h1 + h2 = 0.5 (вихід)", linestyle="--", linewidth=2)
plt.xlabel("h1", fontsize=14, labelpad=6)
plt.ylabel("h2", fontsize=14, labelpad=6)
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.legend(fontsize=12)
plt.gca().set_aspect("equal", adjustable="box")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
