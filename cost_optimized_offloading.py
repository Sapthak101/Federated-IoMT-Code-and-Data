# cost_optimized_offloading.py
import numpy as np

def cost_optimized_offloading(costs, delays, lambd=0.5):
    scores = lambd * costs + (1 - lambd) * delays
    return np.argmin(scores)
