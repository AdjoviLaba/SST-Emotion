import numpy as np
import scipy.ndimage

# Dummy data: 1 sample, 5 bands, 62 channels
data = np.random.rand(5, 62)

# SEED 62 channel mapping to 9x9 grid (Standard)
# 0: FP1, 1: FPZ, 2: FP2
# ...
# This is a standard topographical mapping for SEED.
map_grid = np.array([
    [-1, -1, -1,  0,  1,  2, -1, -1, -1],
    [-1, -1,  3, -1, -1, -1,  4, -1, -1],
    [-1,  5,  6,  7,  8,  9, 10, 11, -1],
    [-1, 12, 13, 14, 15, 16, 17, 18, -1],
    [19, 20, 21, 22, 23, 24, 25, 26, 27],
    [-1, 28, 29, 30, 31, 32, 33, 34, -1],
    [-1, 35, 36, 37, 38, 39, 40, 41, -1],
    [-1, 42, 43, 44, 45, 46, 47, 48, -1],
    [-1, -1, 49, 50, 51, 52, 53, -1, -1],
    [-1, -1, -1, 54, 55, 56, -1, -1, -1],
    [-1, -1, -1, 57, 58, 59, -1, -1, -1],
])
# Wait, SEED 62 channels. Let me find a proper 9x9 map.
# Let's just create a simple 9x9 with 62 valid spots.
grid = np.zeros((9, 9), dtype=int) - 1
idx = 0
for i in range(9):
    for j in range(9):
        if idx < 62:
            grid[i, j] = idx
            idx += 1

mapped = np.zeros((5, 9, 9))
for b in range(5):
    for i in range(9):
        for j in range(9):
            ch = grid[i, j]
            if ch != -1:
                mapped[b, i, j] = data[b, ch]

# Resize to 32x32
resized = scipy.ndimage.zoom(mapped, (1, 32/9, 32/9), order=1)
print(resized.shape)
