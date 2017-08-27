import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("white")
sns.set_context("talk")
fig = plt.figure(1, figsize=(6,4))
fig.clf()
ax = fig.add_subplot(111)

for f in ["loss.dat"]:
    l = np.loadtxt(f)

    ax.plot(np.linspace(0,30, len(l)), l, lw=0.5)
ax.set_yscale("log", nonposy='clip')
sns.despine()

ax.set_xlabel("epoch")
ax.set_ylabel("loss")
plt.tight_layout()
plt.show()
