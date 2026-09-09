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

def distance_sim():
    while True:
        global atoms_p, atoms_v, distance_matrix
        distance_matrix = (np.sum(((atoms_p.reshape(3,1,12)-atoms_p.reshape(3,12,1))**2),axis=0))**(1/2) 
        time.sleep(0.01)
        #like a matrix which contains distances of all 12 atoms with each other pairwise

p_main = thr.Thread(target=simulation)
p_distance = thr.Thread(target=distance_sim)
p_main.start() #threading the process so we can run two functions simaltaneously
p_distance.start()
    

#you can remove the part below its just for testing if the positions are running correctly
#someone please do a matplot for this code
while True: 
    print(distance_matrix)
    time.sleep(1)
#yes

