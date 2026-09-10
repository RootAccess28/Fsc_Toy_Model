import threading as thr #so we can run the simulation process as a thread
import numpy as np
import time as time
n = 12 # generalsing the atom count
cubeSize = 60 #side length of cube
distance_matrix=np.zeros(shape=(n,n))
#treating everthing as m/s
atoms_v = np.random.uniform(low=-10, high=10, size=(n,3)) #the velocity array, we will modify this for simulating things
atoms_a = np.zeros(shape=(n,3)) #acceleration array
atoms_p = np.zeros(shape=(n,3)) #positional array

def simulation():
    global atoms_p, atoms_v
    while True:
        atoms_p=atoms_p+(atoms_v/100) 
        time.sleep(0.01) #updating positions every 10 ms

def distance_sim():
    while True:
        global atoms_p, atoms_v, distance_matrix
        distance_matrix = (np.sum(((atoms_p.reshape(3,1,n)-atoms_p.reshape(3,n,1))**2),axis=0))**(1/2) 
        time.sleep(0.01)
        #like a matrix which contains distances of all 12 atoms with each other pairwise

def cubeBound(): #bounding it inside a cube
    global cubeSize, atoms_v, atoms_p
    while True:
        for x in range(atoms_p.shape[0]):
            for y in range(3):
                if atoms_p[x, y] > cubeSize/2 or atoms_p[x, y] < -cubeSize/2:
                    atoms_v[x,y]=-atoms_v[x,y] #flipping velocities however since there is a time delay of 10ms in the distance simulation , a tiny error may arise in the position

p_main = thr.Thread(target=simulation, daemon=True)
p_distance = thr.Thread(target=distance_sim, daemon=True)
p_bound = thr.Thread(target=cubeBound, daemon=True)
p_main.start() #threading the process so we can run two functions simaltaneously
p_distance.start()
p_bound.start()

#you can remove the part below its just for testing if the positions are running correctly
#someone please do a matplot for this code
while True: 
    print(atoms_p)
    time.sleep(1)
