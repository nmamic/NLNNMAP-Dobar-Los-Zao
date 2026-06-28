import numpy as np

class Dense:
    def __init__(self, in_dim, out_dim, rng):
        limit = np.sqrt(6.0 / (in_dim + out_dim))

        self.W = rng.uniform(
            -limit, limit, size=(in_dim, out_dim)
        ).astype(np.float32)

        self.b = np.zeros(out_dim, dtype=np.float32)

        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad_out):
        self.dW = self.x.T @ grad_out
        self.db = np.sum(grad_out, axis=0)

        grad_x = grad_out @ self.W.T
        return grad_x

    def step(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db


class Tanh:
    def __init__(self):
        self.y = None

    def forward(self, x):
        self.y = np.tanh(x)
        return self.y

    def backward(self, grad_out):
        return grad_out * (1.0 - self.y * self.y)

    def step(self, lr):
        # No parameters.
        pass



class Autoencoder:
    def __init__(self, signal_length=2048, hidden_dim=256, latent_dim=64, rng=None):
        if rng is None:
            rng = np.random.default_rng(0)

        self.layers = [
            Dense(signal_length, hidden_dim, rng),
            Tanh(),

            Dense(hidden_dim, latent_dim, rng),
            Tanh(),

            Dense(latent_dim, hidden_dim, rng),
            Tanh(),

            Dense(hidden_dim, signal_length, rng),
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def step(self, lr):
        for layer in self.layers:
            layer.step(lr)