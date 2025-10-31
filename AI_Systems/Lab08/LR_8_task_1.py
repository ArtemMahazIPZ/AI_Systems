import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


n = 1000
x_data = np.random.rand(n, 1)
noise = np.random.normal(0, 0.5, size=(n, 1))
y_data = 2 * x_data + 1 + noise

k = tf.Variable(tf.random.normal([1]))
b = tf.Variable(tf.zeros([1]))


def model(x):
    return k * x + b


def loss_fn(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

epochs = 2000
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        y_pred = model(x_data)
        loss = loss_fn(y_data, y_pred)
    grads = tape.gradient(loss, [k, b])
    optimizer.apply_gradients(zip(grads, [k, b]))

    if epoch % 200 == 0:
        print(f"Епоха {epoch:4d}: loss={loss.numpy():.4f}, k={k.numpy()[0]:.4f}, b={b.numpy()[0]:.4f}")

print("\nПісля навчання:")
print(f"k ≈ {k.numpy()[0]:.4f}, b ≈ {b.numpy()[0]:.4f}")

plt.scatter(x_data, y_data, s=10, label="Навчальні дані")
plt.plot(x_data, model(x_data), 'r', label=f"y = {k.numpy()[0]:.2f}x + {b.numpy()[0]:.2f}")
plt.title("Навчання лінійної регресії в TensorFlow")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
