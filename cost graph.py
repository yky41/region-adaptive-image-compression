import numpy as np
import matplotlib.pyplot as plt

# candidate bit-depths
b = np.array([3, 4, 5, 6, 7])

# example costs C(S,b)  （示意数据，换成你自己的）
C = np.array([12.5, 9.8, 8.1, 8.6, 10.3])

# find best bit
best_idx = np.argmin(C)
b_star = b[best_idx]
C_star = C[best_idx]

plt.figure(figsize=(6,4))

# discrete points
plt.scatter(b, C, color="black", zorder=3, label="C(S,b)")

# thin line (visual aid, NOT continuous meaning)
plt.plot(b, C, linestyle="--", color="gray", alpha=0.6)

# highlight minimum
plt.scatter(b_star, C_star, color="red", s=80, zorder=4,
            label=r"$b^*(S)=\arg\min_b C(S,b)$")

plt.xlabel("Bit-depth b")
plt.ylabel(r"Total cost $C(S,b)$")
plt.xticks(b)
plt.title(r"Region cost $C(S,b)=R(S,b)+\lambda D(S,b)$")

plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()
