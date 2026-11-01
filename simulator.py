import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Simulation parameters
N = 16          # number of particles
L = 10.0        # box size
dt = 0.01       # time step
rc = 2.0        # cutoff distance
k = 15.0        # spring stiffness
r0 = 0.8        # preferred distance
trail_len = 15  # number of tracer frames

# Initial particle setup
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

def step(pos, vel, mass, dt, L):
    # Velocity Verlet update
    F = pair_forces(pos, rc, k, r0)
    vel += 0.5 * (F / mass[:, None]) * dt
    pos += vel * dt
    pos, vel = walls_reflect(pos, vel, L)
    F_new = pair_forces(pos, rc, k, r0)
    vel += 0.5 * (F_new / mass[:, None]) * dt
    return pos, vel

# Matplotlib setup
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(-0.5, L + 0.5)
ax.set_ylim(-0.5, L + 0.5)
ax.set_aspect('equal')
ax.set_title("Particle Simulation with Tracers")

# Draw box boundary
ax.plot([0, L, L, 0, 0], [0, 0, L, L, 0], 'k-', lw=2)

# Colors for each particle
colors = plt.cm.tab20(np.linspace(0, 1, N))

# Line objects for particle tracers
trails = [ax.plot([], [], '-', color=colors[i], alpha=0.5, lw=1.5)[0] for i in range(N)]
scatter = ax.scatter(pos[:, 0], pos[:, 1], c=colors, s=80, edgecolors='black', zorder=5)

# History buffer to store past positions for tracers
history = []

def update(frame):
    global pos, vel
    pos, vel = step(pos, vel, mass, dt, L)
    
    # Store position history
    history.append(pos.copy())
    if len(history) > trail_len:
        history.pop(0)

    # Update particle tracers
    pts = np.array(history)
    for i in range(N):
        trails[i].set_data(pts[:, i, 0], pts[:, i, 1])

    # Update particle positions
    scatter.set_offsets(pos)
    return [scatter] + trails

anim = FuncAnimation(fig, update, frames=1000, interval=20, blit=True)

if __name__ == "__main__":
    plt.show()
