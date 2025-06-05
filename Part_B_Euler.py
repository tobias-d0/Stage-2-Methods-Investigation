import numpy as np
import matplotlib.pyplot as plt

# Time settings
t_max = 50
h = 0.1
time = np.arange(0, t_max + h, h)

# Initial populations
P_deer_0 = 5
P_squirrel_0 = 23
P_turkey_0 = 35

# Carrying capacities and growth rates
L0 = {'deer': 400, 'squirrel': 4000, 'turkey': 500}
k0 = {'deer': 0.9, 'squirrel': 0.5, 'turkey': 0.32}

# Interaction coefficients
c = {
    'deer': {'squirrel': 0.03, 'turkey': 0.04},
    'squirrel': {'deer': 0.01, 'turkey': 0.04},
    'turkey': {'deer': 0.13, 'squirrel': 0.08},
}
d = {
    'deer': {'squirrel': 0.0002, 'turkey': 0.0004},
    'squirrel': {'deer': 0.0002, 'turkey': 0.0005},
    'turkey': {'deer': 0.0011, 'squirrel': 0.0006},
}

# Function to calculate dP/dt
def dP_dt(P):
    P_deer, P_squirrel, P_turkey = P
    P_dict = {'deer': P_deer, 'squirrel': P_squirrel, 'turkey': P_turkey}
    other_species = {'deer': ['squirrel', 'turkey'], 'squirrel': ['deer', 'turkey'], 'turkey': ['deer', 'squirrel']}
    
    dP = {}
    for sp in ['deer', 'squirrel', 'turkey']:
        j, k = other_species[sp]
        L = L0[sp] - c[sp][j] * P_dict[j] - c[sp][k] * P_dict[k]
        k_val = k0[sp] - d[sp][j] * P_dict[j] - d[sp][k] * P_dict[k]
        dP[sp] = k_val * P_dict[sp] * (1 - P_dict[sp] / L)
    return np.array([dP['deer'], dP['squirrel'], dP['turkey']])

# Euler's Method
P_euler = np.zeros((len(time), 3))
P_euler[0] = [P_deer_0, P_squirrel_0, P_turkey_0]

for i in range(1, len(time)):
    P_euler[i] = P_euler[i-1] + h * dP_dt(P_euler[i-1])

# Plotting
plt.figure(figsize=(12, 6))

# Draw arrows for x and y axis using annotate
plt.annotate('', xy=(max(time)+0.3, 0), xytext=(0, 0),
             arrowprops=dict(arrowstyle="->", lw=1.5, color='black', relpos=(0,0)), annotation_clip=False)

# Y-axis arrow (remains the same)
plt.annotate('', xy=(0, np.max(P_euler)*1.02), xytext=(0, 0),
             arrowprops=dict(arrowstyle="->", lw=1.5, color='black', relpos=(0,0)), annotation_clip=False)

species_names = ['White-tailed Deer', 'Eastern Gray Squirrel', 'Eastern Wild Turkey']
colors = ['green', 'gray', 'brown']

for idx in range(3):
    plt.plot(time, P_euler[:, idx], '-', label=f'{species_names[idx]}', color=colors[idx])
    x_end = time[-1]
    x_prev = time[-2]
    y_end = P_euler[-1, idx]
    y_prev = P_euler[-2, idx]

    # Calculate direction (dx, dy)
    dx = x_end - x_prev
    dy = y_end - y_prev

    # Draw arrow at the end of the curve
    plt.annotate('', xy=(x_end, y_end), xytext=(x_prev, y_prev),
                 arrowprops=dict(arrowstyle='->', color=colors[idx], lw=1.5))
    


plt.xlabel("Time (years)")
plt.ylabel("Population")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xlim(0, t_max)
max_carrying_capacity = max(L0.values())
plt.ylim(0, max_carrying_capacity*1.02) # Add 2% margin
ax = plt.gca()  # Get current axes

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Move left and bottom spine to zero position
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')

# Hide ticks where axes cross
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.show()
