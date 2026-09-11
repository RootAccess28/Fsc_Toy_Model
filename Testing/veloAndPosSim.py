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
pos = np.zeros(shape=(12,3))

def simulation():
    global atoms_p, atoms_v
    while True:
        atoms_p+= atoms_v*dt 
        time.sleep(dt) #updating positions every 10 ms

p_main = thr.Thread(target=simulation) 
p_main.start() #threading the process so we can run two functions simaltaneously

#you can remove the part below its just for testing if the positions are running correctly
#someone please do a matplot for this code
#t
# while True: 
#     print(atoms_p)
#     time.sleep(1)
def update_physics(pos, vel, dt, L):
    pos += vel * dt
    # Simple wall bounce for demonstration
    for dim in range(3):
        mask_low = pos[:, dim] < 0
        mask_high = pos[:, dim] > L
        vel[mask_low, dim] *= -1
        vel[mask_high, dim] *= -1
        pos[mask_low, dim] = 0
        pos[mask_high, dim] = L
    return pos, vel

# Set up the 3D plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Fixed bounds so the camera stays stable
ax.set_xlim([0, L])
ax.set_ylim([0, L])
ax.set_zlim([0, L])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Particle Simulation')

# Initialize scatter plot
scatter = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], color='crimson', s=60, edgecolors='black')

# Animation update callback
def animate(frame):
    global pos, vel
    pos, vel = update_physics(pos, vel, dt, L)
    
    # Efficiently update 3D positions in-place
    scatter._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
    return scatter,

# interval=10 ms matches dt = 0.01s (100 FPS target)
anim = FuncAnimation(fig, animate, frames=200, interval=10, blit=False)

plt.show()


