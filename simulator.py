import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Simulation parameters
N = 16            # number of particles
L = 10.0          # box size
dt = 0.01         # time step
radius = 0.18     # reduced particle radius (cleaner, smaller)
rc = 0.90         # soft repulsion range
k = 28.0          # more intense repulsive stiffness (punchy path deflections)
trail_len = 22    # tracer trail length to see path deflections

# Initial setup: random non-overlapping positions and velocities
np.random.seed(42)
pos = np.random.uniform(radius + 1.0, L - radius - 1.0, size=(N, 2))
vel = np.random.uniform(-7, 7, size=(N, 2))
mass = np.ones(N)

def pair_repulsion(pos, rc, k):
    """Smooth repulsion force that gently bends particle trajectories within distance rc."""
    d = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.sum(d**2, axis=-1))
    
    # Soft repulsion active within range rc
    repel = (dist > 1e-6) & (dist < rc)
    u = d / (dist[:, :, None] + 1e-12)
    fmag = k * (rc - dist) * repel
    return np.sum(fmag[:, :, None] * u, axis=1)

def particle_collisions(pos, vel, radius):
    """Hard elastic core bounce to prevent particles from ever overlapping."""
    min_dist = 2 * radius
    for i in range(N):
        for j in range(i + 1, N):
            delta = pos[i] - pos[j]
            dist = np.hypot(delta[0], delta[1])
            if 0 < dist < min_dist:
                n = delta / dist
                overlap = min_dist - dist
                
                # Separate overlapping cores
                pos[i] += 0.5 * overlap * n
                pos[j] -= 0.5 * overlap * n
                
                # Elastic momentum exchange
                v_rel = vel[i] - vel[j]
                vn = np.dot(v_rel, n)
                if vn < 0:
                    vel[i] -= vn * n
                    vel[j] += vn * n
    return pos, vel

def walls_reflect(pos, vel, L, radius):
    """Reflect particles cleanly off the 4 box walls at their perimeter."""
    for dim in range(2):
        hit_low = pos[:, dim] < radius
        hit_high = pos[:, dim] > (L - radius)
        
        vel[hit_low, dim] = np.abs(vel[hit_low, dim])
        pos[hit_low, dim] = radius
        
        vel[hit_high, dim] = -np.abs(vel[hit_high, dim])
        pos[hit_high, dim] = L - radius
    return pos, vel

def step(pos, vel, mass, dt, L, radius):
    # Velocity Verlet update
    F = pair_repulsion(pos, rc, k)
    vel += 0.5 * (F / mass[:, None]) * dt
    pos += vel * dt
    
    pos, vel = walls_reflect(pos, vel, L, radius)
    pos, vel = particle_collisions(pos, vel, radius)
    
    F_new = pair_repulsion(pos, rc, k)
    vel += 0.5 * (F_new / mass[:, None]) * dt
    return pos, vel

# Matplotlib visualization setup
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(-0.2, L + 0.2)
ax.set_ylim(-0.2, L + 0.2)
ax.set_aspect('equal')
ax.set_title("Particle Simulation with Soft Repulsion & Tracers")

# Draw the box boundaries
ax.plot([0, L, L, 0, 0], [0, 0, L, L, 0], 'k-', lw=2)

# Distinct colors for each particle
colors = plt.cm.tab20(np.linspace(0, 1, N))

# Line objects for trailing path tracers
trails = [ax.plot([], [], '-', color=colors[i], alpha=0.45, lw=1.5)[0] for i in range(N)]

# Circles drawn in exact data units
circles = [
    plt.Circle((pos[i, 0], pos[i, 1]), radius, facecolor=colors[i], edgecolor='black', lw=1.2, zorder=5)
    for i in range(N)
]
for c in circles:
    ax.add_patch(c)

history = []

def update(frame):
    global pos, vel
    pos, vel = step(pos, vel, mass, dt, L, radius)
    
    # Store position history for trailing tracers
    history.append(pos.copy())
    if len(history) > trail_len:
        history.pop(0)

    pts = np.array(history)
    for i in range(N):
        trails[i].set_data(pts[:, i, 0], pts[:, i, 1])
        circles[i].center = (pos[i, 0], pos[i, 1])

    return circles + trails

anim = FuncAnimation(fig, update, frames=1000, interval=20, blit=True)

if __name__ == "__main__":
    plt.show()
