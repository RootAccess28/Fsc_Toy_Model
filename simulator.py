import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Simulation parameters
N = 16          # number of particles
L = 10.0        # box side length
dt = 0.01       # time step
rc = 2.0        # cutoff radius
k = 15.0        # spring constant
r0 = 0.8        # rest length

# Initial conditions
np.random.seed(42)
pos = np.random.uniform(1.0, L - 1.0, size=(N, 2))
vel = np.random.uniform(-4.0, 4.0, size=(N, 2))
mass = np.ones(N)

def pairwise_distances(pos, rc):
    d = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.sum(d**2, axis=-1))
    neighbors = (dist > 0) & (dist <= rc)
    np.fill_diagonal(neighbors, False)
    return d, dist, neighbors

def pair_forces(pos, rc, k, r0):
    d, dist, neighbors = pairwise_distances(pos, rc)
    u = d / (dist[:, :, None] + 1e-12)
    fmag = -k * (dist - r0) * neighbors
    return np.sum(fmag[:, :, None] * u, axis=1)

def walls_reflect(pos, vel, L):
    for dim in range(2):
        hit_low = pos[:, dim] < 0
        hit_high = pos[:, dim] > L
        vel[hit_low, dim] *= -1
        pos[hit_low, dim] = 0.0
        vel[hit_high, dim] *= -1
        pos[hit_high, dim] = L
    return pos, vel

def kinetic_energy(vel, mass):
    return 0.5 * np.sum(mass[:, None] * vel**2)

def potential_energy(pos, rc, k, r0):
    _, dist, neighbors = pairwise_distances(pos, rc)
    upper = np.triu(neighbors, k=1)
    return 0.5 * k * np.sum((dist[upper] - r0)**2)

def step(pos, vel, mass, dt, L):
    # Velocity Verlet (smooth update)
    F = pair_forces(pos, rc, k, r0)
    vel += 0.5 * (F / mass[:, None]) * dt
    pos += vel * dt
    pos, vel = walls_reflect(pos, vel, L)
    F_new = pair_forces(pos, rc, k, r0)
    vel += 0.5 * (F_new / mass[:, None]) * dt
    return pos, vel

# --- Task 1 Numerical Checks ---
d, dist, neighbors = pairwise_distances(pos, rc)
F = pair_forces(pos, rc, k, r0)
KE = kinetic_energy(vel, mass)
PE = potential_energy(pos, rc, k, r0)

print("--- Task 1 Numerical Checks ---")
print("Distance matrix symmetric:", np.allclose(dist, dist.T))
print("Diagonal distances zero:  ", np.allclose(np.diag(dist), 0))
print("Any self-neighbors:       ", np.any(np.diag(neighbors)))
print("Net internal force norm:  ", np.linalg.norm(np.sum(F, axis=0)))
print(f"Energy: K = {KE:.2f}, U = {PE:.2f}, Total E = {KE + PE:.2f}")

# --- Task 2 Matplotlib Animation ---
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-0.5, L + 0.5)
ax.set_ylim(-0.5, L + 0.5)
ax.set_aspect('equal')
ax.set_title("Particle Simulation")
ax.plot([0, L, L, 0, 0], [0, 0, L, L, 0], 'k-', lw=1.5)  # Box outline

scatter = ax.scatter(pos[:, 0], pos[:, 1], c='crimson', s=90, edgecolors='k')

def update(frame):
    global pos, vel
    pos, vel = step(pos, vel, mass, dt, L)
    scatter.set_offsets(pos)
    return scatter,

anim = FuncAnimation(fig, update, frames=600, interval=20, blit=True)

if __name__ == "__main__":
    plt.show()
