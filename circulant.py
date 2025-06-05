import numpy as np
import matplotlib.pyplot as plt

def cov_func(t1, t2, i, j):
    return np.exp(-8 * np.sqrt(
                min(abs(t1[0] - t1[i]), 1 - abs(t1[0] - t1[i]))**2 + 
                min(abs(t2[0] - t2[j]), 1 - abs(t2[0] - t2[j]))**2
            ))

def simulate_circulant(n, cov_func):
    t1 = np.arange(0, 1, 1/n)
    t2 = t1

    G = np.empty((n, n))

    for i in range(n):
        for j in range(n):
            G[i, j] = cov_func(t1, t2, i, j)

    Gamma = np.fft.fft2(G)
    Z = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    X = np.real(np.fft.fft2(np.sqrt(Gamma) * Z / n))

    return X

n = 1 << 8  # 2^8
x = simulate_circulant(n, cov_func)

fig = plt.figure(figsize=(21, 9))
ax1 = plt.subplot(1, 2, 1)

ax1.imshow(x, cmap='gray')

n = 1 << 5
x = simulate_circulant(n, cov_func)

ax2 = plt.subplot(1, 2, 2, projection='3d')
X, Y = np.meshgrid(np.arange(x.shape[1]), np.arange(x.shape[0]))
ax2.plot_surface(X, Y, x, cmap='viridis', alpha=0.7)
plt.tight_layout()
plt.show()
