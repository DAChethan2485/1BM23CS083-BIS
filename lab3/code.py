import numpy as np

def sphere_function(x):
    return np.sum(x**2)

num_dimensions = 2
num_particles =50
max_iterations = 100

w = 0.7
c1 = 1.5
c2 = 1.5

np.random.seed(42)

positions = np.random.uniform(low=-10, high=10, size=(num_particles, num_dimensions))
velocities = np.random.uniform(low=-1, high=1, size=(num_particles, num_dimensions))

pbest_positions = positions.copy()
pbest_scores = np.array([sphere_function(pos) for pos in positions])

gbest_index = np.argmin(pbest_scores)
gbest_position = pbest_positions[gbest_index].copy()
gbest_score = pbest_scores[gbest_index]

for iteration in range(max_iterations):
    for i in range(num_particles):
        fitness = sphere_function(positions[i])
        if fitness < pbest_scores[i]:
            pbest_scores[i] = fitness
            pbest_positions[i] = positions[i].copy()
            if fitness < gbest_score:
                gbest_score = fitness
                gbest_position = positions[i].copy()
    for i in range(num_particles):
        r1 = np.random.rand(num_dimensions)
        r2 = np.random.rand(num_dimensions)
        cognitive_velocity = c1 * r1 * (pbest_positions[i] - positions[i])
        social_velocity = c2 * r2 * (gbest_position - positions[i])
        velocities[i] = w * velocities[i] + cognitive_velocity + social_velocity
        positions[i] = positions[i] + velocities[i]
print(f"max_iteration:{max_iterations}")
print(f"Number of particles:{num_particles}")
print(f"Best position: {gbest_position}")
print(f"Best score: {gbest_score}")
