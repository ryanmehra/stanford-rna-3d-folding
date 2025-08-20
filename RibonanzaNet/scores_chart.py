"""
Owned by Ryan Mehra. Licensed for free use.
Purpose: Plots RMSD and dRMAE scores for model benchmarking.
"""
import matplotlib.pyplot as plt
import numpy as np

# Data
variants = ["Baseline (Linear)", "Transformer + EGNN", "Transformer + RoPE + Bias + EGNN"]
rmsd = [35.3817, 34.5437, 32.2512]
drmae = [4.336786, 3.414121, 2.0878]

# Plot
fig, ax1 = plt.subplots(figsize=(6, 4))

# Create x positions and set a bar width
x = np.arange(len(variants))
width = 0.2  # thinner bars (default is 0.8)

# Bar chart for RMSD with specified width
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # one color per bar
colors = ['#1f77b4', '#1f77b4', '#1f77b4']  # one color per bar
bars = ax1.bar(x, rmsd, width=width, color=colors)
ax1.set_ylim(31, 36)
ax1.set_ylabel("Test RMSD (Å)")
ax1.set_xticks(x)
ax1.set_xticklabels(variants, rotation=45, ha="right")

# Line plot for dRMAE on secondary axis
ax2 = ax1.twinx()
ax2.plot(x, drmae, marker='o', color='orange')
ax2.set_ylabel("dRMAE")

plt.tight_layout()
plt.show()