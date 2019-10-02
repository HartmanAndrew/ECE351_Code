# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 15:15:54 2019

@author: Hartman
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

plt.rcParams.update({'font.size': 14})
steps = 0.01
t = np.arange(-10,10+steps,steps)


def step(t):
    y = np.zeros((t.shape))
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else :
            y[i] = 1
    return y

def ramp(t):
    y = np.zeros((t.shape))
    for i in range(len(t)):
        if t[i] >= 0:
            y[i] = t[i]
        else:
            y[i] = 0
    return y

def h1(t):
    return np.exp(2*t) * step(1-t)
def h2(t):
    return step(t-2)-step(t-6)
def h3(t):
    return np.cos(2*np.pi*0.25*t)*step(t) #omega = 2*pi*f

def my_Convolve(f1, f2):
    LenF1 = len(f1)
    LenF2 = len(f2)
    f1Extended = np.append(f1, np.zeros((1,LenF2-1))) #Extend the functions so that they are the same size
    f2Extended = np.append(f2, np.zeros((1,LenF1-1)))
    result = np.zeros(f1Extended.shape) #Create a result array of the same length
    for i in range(LenF2+LenF1-2): #loop through the array
        result[i] = 0
        for j in range(LenF1): #Loop through the first array and multiply the results together adding them to result
            if(i-j+1>0):
                try:
                    result[i] = result[i]+f1Extended[j]*f2Extended[i-j+1]
                except:
                    print(i,j)
    return result


myFigSize = (10,15)

# Plotting the 3 functions
plt.figure(figsize=myFigSize)

plt.subplot(3,1,1)
plt.plot(t,h1(t))
plt.grid(True)
plt.ylabel('y(t)')
plt.title('h1(t)')


plt.subplot(3,1,2)
plt.plot(t,h2(t))
plt.grid(True)
plt.ylabel('y(t)')
plt.title('h2(t)')


plt.subplot(3,1,3)
plt.plot(t,h3(t))
plt.grid(True)
plt.ylabel('y(t)')
plt.title('h3(t)')



t2 = np.arange(2*t[0], 2*t[len(t)-1]+steps, steps)


plt.figure(figsize=myFigSize)
plt.subplot(3,1,1)
plt.plot(t2,my_Convolve(h1(t), step(t)) * steps)
plt.grid(True)
plt.ylabel('y')
plt.title('Step reponse of h1(t)')



plt.subplot(3,1,2)
plt.plot(t2,my_Convolve(h2(t), step(t)) * steps)
plt.grid(True)
plt.ylabel('y(t)')
plt.title('Step reponse of h2(t)')


plt.subplot(3,1,3)
plt.plot(t2,my_Convolve(h3(t), step(t)) * steps)
plt.grid(True)
plt.ylabel('y(t)')
plt.title('Step reponse of h3(t)')



t = np.arange(-20, 20+steps, steps)
plt.figure(figsize=myFigSize)
plt.subplot(3,1,1)
plt.plot(t, 0.5* (np.exp(2)*step(t-1) + np.exp(2*t)*step(1-t)))
plt.grid(True)
plt.ylabel('y(t)')
plt.title('Step Response found by hand of h1(t)')

plt.subplot(3,1,2)
plt.plot(t,((t-2)*step(t-2)) - (t-6)*step(t-6))
plt.grid(True)
plt.ylabel('y(t)')
plt.title('Step Response found by hand of h2(t)')

plt.subplot(3,1,3)
plt.plot(t,0.6366197724*np.sin(1.570796327*t)*step(t))
plt.grid(True)
plt.ylabel('y(t)')
plt.title('Step Response found by hand of h3(t)')



plt.show()