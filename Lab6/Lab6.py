# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 19:14:40 2019

@author: Hartman
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

plt.rcParams.update({'font.size': 14})
steps = 1e-3
t = np.arange(0,2+steps,steps)

def step(t):
    y = np.zeros((t.shape))
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else :
            y[i] = 1
    return y
def cosineMethod(r, p, t):
    y = np.zeros((t.shape))
    for i in range(len(r)):
        k = np.abs(r[i])
        phi = np.angle(r[i])
        preal = p[i].real
        pimag = p[i].imag
        y = y+(k*np.exp(preal * t) * np.cos((pimag * t) + phi) * step(t))
    
    return y

##----------------------Part 1 ----------------------##
y = (0.5+np.exp(-6*t)-0.5*np.exp(-4*t))*step(t)

numH = [1,6,12] # Ls + 0 - Numerator
denH = [1,10,24] #RCLs + Ls + R - Denominator

tout, yout = sig.step((numH,denH), T=t)

myFigSize = (10,15)

plt.figure(figsize=myFigSize)

plt.subplot(2,1,1)
plt.plot(t,y)
plt.grid(True)
plt.ylabel('y(t)')
plt.title('Hand Solved y(t)')

plt.subplot(2,1,2)
plt.plot(tout, yout)
plt.grid(True)
plt.ylabel('y(t)')
plt.title('y(t) using sig.step()')

plt.show();
denH = [1,10,24,0]

print('Part 1:')
r,p,_ = sig.residue(numH, denH, tol=1e-3)
print('r='+str(r))
print('')
print('p='+str(p))




##-----------------------------Part 2 -----------------------------------##
print('')
print('Part 2:')
numH = [0,0,0,0,0,25250]
denH = [1,18,218,2036,9085,25250,0]
r,p,_ = sig.residue(numH, denH, tol=1e-3)
print('r='+str(r))
print('')
print('p='+str(p))

y = cosineMethod(r,p,t)
plt.figure(figsize=myFigSize)
plt.subplot(2,1,1)
plt.plot(t, y)
plt.grid(True)
plt.ylabel('y(t)')
plt.title('y(t) using cosine method')


numH = [0,0,0,0,0,25250]
denH = [1,18,218,2036,9085,25250]
# Part 2-3
tout, yout = sig.step((numH,denH), T=t)
plt.subplot(2,1,2)
plt.plot(tout, yout)
plt.grid(True)
plt.ylabel('y(t)')
plt.title('y(t) using scipy.step')

plt.show();
