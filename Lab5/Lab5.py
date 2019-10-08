# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 19:13:54 2019

@author: Hartman
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

plt.rcParams.update({'font.size': 14})
steps = 1e-5
t = np.arange(0,0.0012+steps,steps)


def step(t):
    y = np.zeros((t.shape))
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else :
            y[i] = 1
    return y

def deg2rad(angle):
    return (np.pi*angle)/180

myFigSize = (10,15)

# Plotting the 3 functions
plt.figure(figsize=myFigSize)

R = 1000
L = 0.027
C = 100e-9
#******************PART 1********************
numH = [0, L,0] # Ls + 0 - Numerator
denH = [R*C*L, L, R] #RCLs + Ls + R - Denominator

tout, yout = sig.impulse((numH,denH), T=t)

HandSolved_ht = (10356)*np.exp(-5000*t)*np.sin(18584*t+deg2rad(105))*step(t)

plt.subplot(2,1,1)
plt.plot(t,HandSolved_ht)
plt.grid(True)
plt.ylabel('y(t)')
plt.title('h(t) Hand Solved')


plt.subplot(2,1,2)
plt.plot(tout, yout)
plt.grid(True)
plt.ylabel('y(t)')
plt.title('h(t) using sig.impulse()')
#*********************Part 1 End ***************

#********************Part 2***********************
myFigSize = (10,7)
plt.figure(figsize=myFigSize)
toutstep, youtstep = sig.step((numH,denH), T=t)
plt.subplot(1,1,1)
plt.plot(toutstep,youtstep)
plt.grid(True)
plt.ylabel('y(t)')
plt.title('step response')

plt.show();