# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 19:15:51 2019

@author: Hartman
"""

#Equation y{t} = r(t)-r(t-3)+5u(t-3)-2u(t-6)-2r(t-6)


import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14}) # set font size in plots
steps = 0.01
t = np.arange(-5,10+steps,steps) # to go up to 5.0, we must add a stepSize since `np.arange()`
 # goes up to (without including) the value of the second argument
#print('# of elements: len(t) =',len(t), # notice this may be one larger than expected since `t` starts at 0
# '\nFirst element: t[0] =',t[0], # index the first value of the array `t`
# '\nLast element: t[len(t)-1] =',t[len(t)-1]) # index the last value of the array `t`


def func1(t):
    y=np.zeros((len(t),1))
    for i in range(len(t)):
        y[i] = np.cos(t[i])
    return y


def step(t):
    y = np.zeros((len(t),1))
    for i in range(len(t)):
        if t[i] > 0:
            y[i] = 1
        else :
            y[i] = 0
    return y

def ramp(t):
    y = np.zeros((len(t),1))
    for i in range(len(t)):
        if t[i] > 0:
            y[i] = t[i]
        else:
            y[i] = 0
    return y
    
def func_stepRamp(t):
    func = (ramp(t))-(ramp(t-3))+(5*step(t-3))-(2*step(t-6))-(2*ramp(t-6))
    return func





myFigSize = (10,8)

# =============================================================================
# plt.figure(figsize=myFigSize)
# plt.subplot(1,1,1)
# plt.plot(t,func1(t))
# plt.grid(True)
# plt.ylabel('y(t)')
# plt.title('Part1 Task2')
# plt.xlabel('t')
# plt.show()
# =============================================================================



stepFunc = step(t-2)
rampFunc = ramp(t-2)
func = func_stepRamp(t)
plt.figure(figsize=myFigSize)
plt.subplot(3,1,1)
plt.plot(t,stepFunc)
plt.grid(True)
plt.ylabel('y(t)')
plt.title('StepFunc')

plt.subplot(3,1,2)
plt.plot(t,rampFunc)
plt.grid(True)
plt.ylabel('y(t)')
plt.title('rampFunc')

plt.subplot(3,1,3)
plt.plot(t,func)
plt.grid(True)
plt.ylabel('y(t)')
plt.title('Part2Task3')


#Time Shifting
t = np.arange(-15,15+steps,steps)
funcTimeReverse = func_stepRamp(-t)
myFigSize2= (10,30)
plt.figure(figsize=myFigSize2)
plt.subplot(6,1,1)
plt.plot(t,funcTimeReverse)
plt.grid(True)
plt.ylabel('y(t)')
plt.title('funcTimeReverse - -t')

funcTask2 = func_stepRamp(t-4)
plt.subplot(6,1,2)
plt.plot(t,funcTask2)
plt.grid(True)
plt.ylabel('y(t)')
plt.title('func Task2 - t-4')

funcTask2_2 = func_stepRamp(-t-4)
plt.subplot(6,1,3)
plt.plot(t,funcTask2_2)
plt.grid(True)
plt.ylabel('y(t)')
plt.title('func Task2_2 - -t-4')

funcTask3 = func_stepRamp(t / 2)
plt.subplot(6,1,4)
plt.plot(t,funcTask3)
plt.grid(True)
plt.ylabel('y(t)')
plt.title('func Task3 - t/2')

funcTask3_2 = func_stepRamp(2*t)
plt.subplot(6,1,5)
plt.plot(t,funcTask3_2)
plt.grid(True)
plt.axis([-15,10,-10,10])
plt.ylabel('y(t)')
plt.title('func Task3_2 - 2*t')


# =============================================================================
# steps = 0.5
# t = np.arange(-15,15+steps,steps)
# =============================================================================
#Derivative of function
dt = np.diff(t)
funcTask5 = np.diff(func_stepRamp(t),axis=0)/dt


plt.figure(figsize=myFigSize2)
plt.subplot(6,1,1)
plt.plot(t[0:len(t)-1],funcTask5,label='F\'(t)')
plt.grid(True)
plt.ylabel('y\'(t)')
plt.axis([-1,10,-4,10])
plt.title('func Task5 - Derivative')










plt.xlabel('t')
plt.show()       

 