import numpy as np
import matplotlib.pyplot as plt

def simulate_ma_gaussian(n, r):
    noise = np.random.randn(n, n)
    x_grid, y_grid = np.meshgrid(np.arange(-r, r+1), np.arange(-r, r+1))
    mask = (x_grid**2 + y_grid**2) <= r**2
    x = np.zeros((n, n))
    nmin, nmax = r ** 2 + 1, n - r ** 2 - 1

    for i in range(nmin, nmax+1):
        for j in range(nmin, nmax+1):
            A = noise[(i-r) : (i+r+1), (j-r) : (j+r+1)]
            x[i, j] = np.sum(A * mask)

    Nr = np.sum(mask)
    x = x[nmin:nmax+1, nmin:nmax+1] / Nr

    return x

fig = plt.figure(figsize=(21, 9))
ax1 = plt.subplot(1, 2, 1)

x1 = simulate_ma_gaussian(300, 6)

ax1.imshow(x1, cmap='gray')

x2 = simulate_ma_gaussian(50, 4)

ax2 = plt.subplot(1, 2, 2, projection='3d')
X, Y = np.meshgrid(np.arange(x2.shape[1]), np.arange(x2.shape[0]))
ax2.plot_surface(X, Y, x2, cmap='viridis')
plt.tight_layout()
plt.show()

