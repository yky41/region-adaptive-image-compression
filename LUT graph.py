import matplotlib.pyplot as plt
import numpy as np

x = np.arange(256)
y = x // 32   # b=3 LUT

plt.figure(figsize=(8, 4))
plt.step(x, y, where="post")
plt.yticks(range(8))
plt.xticks(range(0, 257, 32))
plt.xlabel("Input intensity (0–255)")
plt.ylabel("Quantized level (0–7)")
plt.title("LUT for b = 3 (8 levels, exact 32-to-1 mapping)")
plt.grid(True, axis="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
