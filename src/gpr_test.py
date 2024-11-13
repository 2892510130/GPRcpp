import numpy as np
import matplotlib.pyplot as plt

print("Loading data...")
data = np.loadtxt("C:\\Users\\pc\\Desktop\\Personal\\Code\\KnowledgeGallery\\Machine Learning Library\\GPRcpp\\Log\\gpr.txt")

plt.plot(data[:, 0], label = "p_w")
plt.plot(data[:, 1], label = "p_v")
plt.plot(data[:, 2], label = "r_w")
plt.plot(data[:, 3], label = "r_v")
plt.legend()
plt.show()