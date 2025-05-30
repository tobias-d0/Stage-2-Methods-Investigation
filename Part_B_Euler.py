import numpy as np
import matplotlib.pyplot as plt

# Time settings
t_max = 50
h = 0.1
time = np.arange(0, t_max + h, h)

# Initial populations
P_deer_0 = 20
P_squirrel_0 = 200
P_turkey_0 = 60

# Carrying capacities and growth rates
L0 = {'deer': 400, 'squirrel': 4000, 'turkey': 300}
k0 = {'deer': 0.9, 'squirrel': 0.5, 'turkey': 0.12}

# Interaction coefficients
c = {
    'deer': {'squirrel': 0.01, 'turkey': 0.1},
    'squirrel': {'deer': 0.05, 'turkey': 0.03},
    'turkey': {'deer': 0.03, 'squirrel': 0.01},
}
d = {
    'deer': {'squirrel': 0.0001, 'turkey': 0.001},
    'squirrel': {'deer': 0.0003, 'turkey': 0.0002},
    'turkey': {'deer': 0.0002, 'squirrel': 0.0001},
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

species_names = ['White-tailed Deer', 'Eastern Gray Squirrel', 'Eastern Wild Turkey']
colors = ['green', 'gray', 'brown']

for idx in range(3):
    plt.plot(time, P_euler[:, idx], '-', label=f'{species_names[idx]}', color=colors[idx])

plt.xlabel("Time (years)")
plt.ylabel("Population")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
