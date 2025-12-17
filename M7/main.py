import numpy as np
import matplotlib.pyplot as plt
import time
import warnings

warnings.filterwarnings("ignore")

print("Gas Simulation Parameters")
while True:
    try:
        N_input = input("Enter number of particles (50-2000) [default=500]: ").strip()
        N = 500 if N_input == "" else int(N_input)
        if 50 <= N <= 2000:
            break
        print("Number must be between 50 and 2000")
    except ValueError:
        print("Invalid input. Please enter a number.")

while True:
    try:
        time_input = input("Enter simulation time in seconds (1.0-10.0) [default=3.0]: ").strip()
        total_time = 3.0 if time_input == "" else float(time_input)
        if 1.0 <= total_time <= 10.0:
            break
        print("Time must be between 1.0 and 10.0 seconds")
    except ValueError:
        print("Invalid input. Please enter a number.")

m = 1.0
g = 0.1
radius = 0.01
width = 1.0
height = 2.0
v0 = 0.5
dt = 0.001
equilibrium_time = max(1.0, total_time * 0.7)
num_steps = int(total_time / dt)

cell_size = 4 * radius
num_cells_x = int(width / cell_size) + 2
num_cells_y = int(height / cell_size) + 2

np.random.seed(42)
positions = np.zeros((N, 2))
velocities = np.zeros((N, 2))

grid_cols = int(np.sqrt(N * width / height)) + 1
grid_rows = (N + grid_cols - 1) // grid_cols

x_spacing = width / (grid_cols + 1)
y_spacing = 0.1 / (grid_rows + 1)

idx = 0
for i in range(grid_rows):
    for j in range(grid_cols):
        if idx >= N:
            break
        x_pos = (j + 1) * x_spacing
        y_pos = (i + 1) * y_spacing + radius
        x_pos = max(radius, min(x_pos, width - radius))
        y_pos = max(radius, min(y_pos, 0.1))
        positions[idx] = [x_pos, y_pos]
        idx += 1

theta = 2 * np.pi * np.random.rand(N)
velocities[:, 0] = v0 * np.cos(theta)
velocities[:, 1] = v0 * np.sin(theta)

initial_ke = 0.5 * m * np.sum(velocities ** 2)
initial_pe = m * g * np.sum(positions[:, 1])
initial_energy = initial_ke + initial_pe

start_time = time.time()

for step in range(num_steps):
    velocities[:, 1] -= 0.5 * g * dt
    positions += velocities * dt

    left_mask = positions[:, 0] < radius
    right_mask = positions[:, 0] > width - radius
    velocities[left_mask, 0] *= -1
    velocities[right_mask, 0] *= -1
    positions[left_mask, 0] = radius
    positions[right_mask, 0] = width - radius

    bottom_mask = positions[:, 1] < radius
    top_mask = positions[:, 1] > height - radius
    velocities[bottom_mask, 1] *= -1
    velocities[top_mask, 1] *= -1
    positions[bottom_mask, 1] = radius
    positions[top_mask, 1] = height - radius

    grid = [[[] for _ in range(num_cells_y)] for _ in range(num_cells_x)]

    for i in range(N):
        x, y = positions[i]
        cell_x = int(x / cell_size)
        cell_y = int(y / cell_size)
        cell_x = max(0, min(cell_x, num_cells_x - 1))
        cell_y = max(0, min(cell_y, num_cells_y - 1))
        grid[cell_x][cell_y].append(i)

    for i in range(N):
        x_i, y_i = positions[i]
        cell_x_i = int(x_i / cell_size)
        cell_y_i = int(y_i / cell_size)
        cell_x_i = max(0, min(cell_x_i, num_cells_x - 1))
        cell_y_i = max(0, min(cell_y_i, num_cells_y - 1))

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx = int(cell_x_i + dx)
                ny = int(cell_y_i + dy)
                if nx < 0 or nx >= num_cells_x or ny < 0 or ny >= num_cells_y:
                    continue
                for j in grid[nx][ny]:
                    if i >= j:
                        continue

                    dx_ij = positions[j, 0] - positions[i, 0]
                    dy_ij = positions[j, 1] - positions[i, 1]
                    distance_sq = dx_ij * dx_ij + dy_ij * dy_ij
                    min_distance = 2 * radius
                    min_dist_sq = min_distance * min_distance

                    if distance_sq < min_dist_sq:
                        if distance_sq < 1e-15:
                            angle = np.random.uniform(0, 2 * np.pi)
                            nx_dir, ny_dir = np.cos(angle), np.sin(angle)
                            distance = 1e-8
                        else:
                            distance = np.sqrt(distance_sq)
                            nx_dir = dx_ij / distance
                            ny_dir = dy_ij / distance

                        dvx = velocities[i, 0] - velocities[j, 0]
                        dvy = velocities[i, 1] - velocities[j, 1]
                        dot_product = dvx * nx_dir + dvy * ny_dir

                        velocities[i, 0] -= dot_product * nx_dir
                        velocities[i, 1] -= dot_product * ny_dir
                        velocities[j, 0] += dot_product * nx_dir
                        velocities[j, 1] += dot_product * ny_dir

                        if distance < min_distance:
                            overlap = (min_distance - distance) * 0.5
                            positions[i, 0] -= overlap * nx_dir
                            positions[i, 1] -= overlap * ny_dir
                            positions[j, 0] += overlap * nx_dir
                            positions[j, 1] += overlap * ny_dir

    velocities[:, 1] -= 0.5 * g * dt

    if step % 500 == 0 and step > 0:
        elapsed = time.time() - start_time
        remaining = elapsed * (num_steps - step) / step
        print(f"Step {step}/{num_steps} | "
              f"Elapsed: {elapsed:.1f}s | "
              f"ETA: {remaining:.1f}s")

print(f"Simulation completed in {time.time() - start_time:.1f} seconds")

speeds = np.sqrt(velocities[:, 0] ** 2 + velocities[:, 1] ** 2)
y_positions = positions[:, 1]

v_avg_sq = np.mean(speeds ** 2)
kT = m * v_avg_sq / 2

v_theory = np.linspace(0, np.max(speeds) * 1.2, 200)
f_theory_speed = (m / kT) * v_theory * np.exp(-m * v_theory ** 2 / (2 * kT))

h_theory = np.linspace(0, height, 200)
f_theory_height = np.exp(-m * g * h_theory / kT)
area = np.trapz(f_theory_height, h_theory)
f_theory_height /= area

hist_height, bin_edges = np.histogram(y_positions, bins=50, range=(0, height), density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.hist(speeds, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black', label='Simulation')
plt.plot(v_theory, f_theory_speed, 'r-', linewidth=2.5, label='2D Maxwell-Boltzmann')
plt.xlabel('Speed', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.title('Speed Distribution at Equilibrium', fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.bar(bin_centers, hist_height, width=bin_edges[1] - bin_edges[0],
        alpha=0.7, color='lightgreen', edgecolor='black', label='Simulation')
plt.plot(h_theory, f_theory_height, 'b-', linewidth=2.5, label='Barometric Formula')
plt.xlabel('Height', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.title('Height Distribution at Equilibrium', fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('gas_simulation_results.png', dpi=300)
plt.show()

KE = 0.5 * m * np.sum(velocities ** 2, axis=1)
PE = m * g * positions[:, 1]
total_energy = np.sum(KE + PE)

print(f"\nEnergy Conservation Check:")
print(f"Initial total energy: {initial_energy:.4f}")
print(f"Final total energy:   {total_energy:.4f}")
print(f"Relative change:      {abs(total_energy - initial_energy) / initial_energy:.2%}")
print(f"Average kinetic energy per particle: {np.mean(KE):.4f}")
print(f"Temperature parameter (kT): {kT:.4f}")
print(f"Mean height: {np.mean(y_positions):.4f}")

if __name__ == "__main__":
    main()
