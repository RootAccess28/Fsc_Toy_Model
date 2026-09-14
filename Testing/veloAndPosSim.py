import threading as thr #so we can run the simulation process as a thread
import numpy as np
import time as time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


#treating everthing as m/s
N=12
L=10.0
dt=0.01        # Creating a variable for storing the least count of time
vel = np.random.uniform(low=-10, high=10, size=(12,3))    #the velocity array, we will modify this for simulating things
pos = np.random.uniform(0.5, L - 0.5, size=(N, 3))
radius=0.1

# def simulation():
#     global atoms_p, atoms_v
#     while True:
#         atoms_p+= atoms_v*dt 
#         time.sleep(dt) #updating positions every 10 ms

# p_main = thr.Thread(target=simulation) 
# p_main.start() #threading the process so we can run two functions simaltaneously

def update_physics(pos, vel, dt, L):
    pos += vel * dt
    # Simple wall bounce for demonstration
    for dim in range(3):
        mask_low = pos[:, dim] < (0+radius)
        mask_high = pos[:, dim] > (L-radius)
        vel[mask_low, dim] *= -1
        vel[mask_high, dim] *= -1
        pos[mask_low, dim] = (0+radius)
        pos[mask_high, dim] = (L-radius)
    return pos, vel


def draw_cube_edges(ax, L):
    # 8 vertices of the box [0, L]^3
    r = [0, L]
    # Draw line segments along all 12 edges
    for x in r:
        for y in r:
            ax.plot([x, x], [y, y], [0, L], color='gray', linestyle='--', linewidth=0.8)
            ax.plot([x, x], [0, L], [y, y], color='gray', linestyle='--', linewidth=0.8)
            ax.plot([0, L], [x, x], [y, y], color='gray', linestyle='--', linewidth=0.8)


# Set up the 3D plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Fixed bounds so the camera stays stable
ax.set_xlim([0, L])
ax.set_ylim([0, L])
ax.set_zlim([0, L])
ax.set_box_aspect([1, 1, 1])


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Particle Simulation')


draw_cube_edges(ax,L)

# Prevent distortion so the box is an exact cube
ax.set_box_aspect([1, 1, 1])

# Set a higher, wider initial camera angle (elevation, azimuth)
ax.view_init(elev=25, azim=45)

# Initialize scatter plot
scatter = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], color='crimson', s=60, edgecolors='black')

# Animation update callback
def animate(frame):
    global pos, vel
    pos, vel = update_physics(pos, vel, dt, L)
    
    # Efficiently update 3D positions in-place
    scatter._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])

    ax.view_init(elev=20, azim=(45 + frame * 0.4) % 720)
    return scatter,

# interval=10 ms matches dt = 0.01s (100 FPS target)
anim = FuncAnimation(fig, animate, frames=2000, interval=10, blit=False)

plt.show()


