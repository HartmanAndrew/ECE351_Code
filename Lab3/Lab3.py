# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 19:30:06 2019

@author: Hartman
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

plt.rcParams.update({'font.size': 14})
steps = 0.01
t = np.arange(0,20+steps,steps)

def step(t):
    y = np.zeros((len(t)))
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else :
            y[i] = 1
    return y

def ramp(t):
    y = np.zeros((len(t)))
    for i in range(len(t)):
        if t[i] >= 0:
            y[i] = t[i]
        else:
            y[i] = 0
    return y

def func_1(t):
    func = (step(t-2) - step(t-9))
    return func
def func_2(t):
    func = (np.exp(-t))
    return func
def func_3(t):
    #func = np.zeros((len(t)))
    func = (ramp(t-2)*(step(t-2)-step(t-3)) + ramp(4-t)*(step(t-3)-step(t-4)))
    return func



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
plt.plot(t,func_1(t))
plt.grid(True)
plt.ylabel('y(t)')
plt.title('f1(t)')

plt.subplot(3,1,2)
plt.plot(t,func_2(t))
plt.grid(True)
plt.ylabel('y(t)')
plt.title('f2(t)')

plt.subplot(3,1,3)
plt.plot(t,func_3(t))
plt.grid(True)
plt.ylabel('y(t)')
plt.title('f3(t)')


#Part 2 Plotting My convolution

t2 = np.arange(0,40+3*steps,steps)
plt.figure(figsize=myFigSize)
plt.subplot(3,1,1)
plt.plot(t2,my_Convolve(func_1(t), func_2(t)) * steps)
plt.grid(True)
plt.ylabel('y(t)')
plt.title('f1(t) * f2(t)')

plt.subplot(3,1,2)
plt.plot(t2,my_Convolve(func_2(t), func_3(t)) * steps)
plt.grid(True)
plt.ylabel('y(t)')
plt.title('f2(t) * f3(t)')

plt.subplot(3,1,3)
plt.plot(t2,my_Convolve(func_1(t), func_3(t)) * steps)
plt.grid(True)
plt.ylabel('y(t)')
plt.title('f1(t) * f3(t)')

func1 = func_1(t)
func2 = func_2(t)
func_scipy = sig.convolve(func1,func2)

# Plotting the SciPy Convolutions

plt.figure(figsize=myFigSize)
plt.subplot(3,1,1)
plt.plot(t2,sig.convolve(func_1(t),func_2(t))*steps)
plt.grid(True)
plt.ylabel('y(t)')
plt.title('f1(t) * f2(t)')

plt.subplot(3,1,2)
plt.plot(t2,sig.convolve(func_2(t),func_3(t))*steps)
plt.grid(True)
plt.ylabel('y(t)')
plt.title('f2(t) * f3(t)')

plt.subplot(3,1,3)
plt.plot(t2,sig.convolve(func_1(t),func_3(t))*steps)
plt.grid(True)
plt.ylabel('y(t)')
plt.title('f1(t) * f3(t)')

plt.xlabel('t')
plt.show()

