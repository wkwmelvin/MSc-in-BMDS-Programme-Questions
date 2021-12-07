import numpy as np
import matplotlib.pyplot as plt 

def runge_katta_4(fn_list, a, b, c, d, t, dt):
    k1 = []
    k2 = []
    k3 = []
    k4 = []
    
    for fn in fn_list:
        k1.append(dt * fn(t, a, b, c, d))
        
    for fn in fn_list:
        k2.append(dt * fn(t + dt/2, a + k1[0]/2, b + k1[1]/2, c + k1[2]/2, d + k1[3]/2))
        
    for fn in fn_list:
        k3.append(dt * fn(t + dt/2, a + k2[0]/2, b + k2[1]/2, c + k2[2]/2, d + k2[3]/2))
        
    for fn in fn_list:
        k4.append(dt * fn(t + dt/2, a + k3[0], b + k3[1], c + k3[2], d + k3[3]/2))
    
    k1 = np.array(k1)
    k2 = np.array(k2)
    k3 = np.array(k3)
    k4 = np.array(k4)

    output = (1/6) * (k1 + k2 + k3 + k4)
    
    return output

# the four equations
k_1 = 100
k_2 = 600
k_3 = 150

def dP(t, E, S, ES, P):
    return (k_3 * ES)

def dES(t, E, S, ES, P):
    return (k_1 * E * S) - (k_2 * ES) - (k_3 * ES)

def dE(t, E, S, ES, P):
    return (k_2 * ES) + (k_3 * ES) - (k_1 * E * S)

def dS(t, E, S, ES, P):
    return (k_3 * ES) - (k_1 * E * S)

# set initial values
ES = [0]
P = [0]
S = [10]
E = [1]
time = []

fn_list = [dE, dS, dES, dP]
dt = 0.001
t = 0
time_end = 0.1
iter_no = 0

#iterate for step size dt until desired end time
while t < time_end:
    time.append(t)
    t += dt
    output = runge_katta_4(fn_list, E[iter_no], S[iter_no], ES[iter_no], P[iter_no], t, dt)
    
    
    E.append(E[iter_no] + output[0])
    S.append(S[iter_no] + output[1])
    ES.append(ES[iter_no] + output[2])
    P.append(P[iter_no] + output[3])
    
    iter_no += 1

time.append(time_end)

#plot the concentration against time
plt.plot(time, E, label = 'E')
plt.plot(time, S, label = 'S')
plt.plot(time, ES, label = 'ES')
plt.plot(time, P, label = 'P')
plt.xlabel('Time (min)')
plt.ylabel('Concentratioon (micro M)')
plt.legend()
plt.title('Concentration against Time')


#plot velocity against S
V = k_3 * np.array(ES)

plt.figure()
plt.plot(S, V)
plt.xlabel('Concentration of S (micro M)')
plt.ylabel('Velocity (micro M per min)')
plt.title('Velocity against Concentration of S')

