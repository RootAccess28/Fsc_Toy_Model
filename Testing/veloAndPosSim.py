import threading as thr #so we can run the simulation process as a thread
import numpy as np
import time as time
#treating everthing as m/s
atoms_v = np.random.uniform(low=-10, high=10, size=(12,3)) #the velocity array, we will modify this for simulating things
atoms_p = np.zeros(shape=(12,3))

def simulation():
    global atoms_p, atoms_v
    while True:
        atoms_p=atoms_p+(atoms_v/100) 
        time.sleep(0.01) #updating positions every 10 ms

p_main = thr.Thread(target=simulation) 
p_main.start() #threading the process so we can run two functions simaltaneously

while True:
    print(atoms_p)
    time.sleep(1)


