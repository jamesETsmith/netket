# Load the data from the .log file
import json
import matplotlib.pyplot as plt
import numpy as np

# filename = "H4/H4.log"
# filename = "H6/H6.log"
filename = "C2/C2.log"
# filename = "Li2.log"
N = 20  # Number of point to average over


data = json.load(open(filename))

# Extract the relevant information

iters = []
energy = []
sf = []
sigma = []
var = []
var_sigma = []
accept = []


for iteration in data["Output"]:
    iters.append(iteration["Iteration"])
    accept.append(iteration["Acceptance"])
    energy.append(iteration["Energy"]["Mean"])
    sigma.append(iteration["Energy"]["Sigma"])
    var.append(iteration["EnergyVariance"]["Mean"])
    var_sigma.append(iteration["EnergyVariance"]["Sigma"])


fig, axes = plt.subplots(2, 1)
ax1 = axes[0]
ax2 = axes[1]

#
# Energy Axis
#
ax1.errorbar(iters, energy, fmt="o", yerr=sigma, color="blue", label="Energy")
ax1.set_ylabel("Energy")
ax1.set_xlabel("Iteration")
ax1.legend(loc=2)


# Inset
axins = ax1.inset_axes([0.5, 0.5, 0.47, 0.47])
axins.errorbar(
    iters[-N:], energy[-N:], fmt="o", yerr=sigma[-N:], color="blue", label="Energy"
)
axins.set_ylabel("Energy")
axins.set_xlabel("Iteration")

#
# Variance Axis
#
ax2.errorbar(iters, var, fmt="o", yerr=var_sigma, color="red", label="EnergyVariance")
ax2.set_ylabel("Variance")
ax2.set_xlabel("Iteration")
ax2.legend(loc=2)

axins2 = ax2.inset_axes([0.5, 0.5, 0.47, 0.47])
axins2.errorbar(
    iters[-N:],
    var[-N:],
    fmt="o",
    yerr=var_sigma[-N:],
    color="red",
    label="EnergyVariance",
)
axins2.set_ylabel("EnergyVariance")
axins2.set_xlabel("Iteration")
axins2.set_ylim([-1.1 * abs(np.mean(var[-N:])), 1.2 * abs(np.mean(var[-N:]))])

# ax2p = ax2.twinx()
# ax2p.plot(iters, accept, "^", label="Acceptance")
# ax2p.legend(loc=3)

#
print(np.mean(energy[-N:]))
# plt.show()
plt.tight_layout()
plt.savefig(filename[:-3] + "png")
