# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 19:43:53 2019

@author: Hartman
"""

import numpy as np
import matplotlib.pyplot as plt
# =============================================================================
# t = 1
# print(t)
# print("t =", t)
# print('t =', t,"seconds\n and t is not minutes")
# print(3**2)
# =============================================================================

#=============================================================================
#Arrays can be defined in multiple ways, either using the default python lists
#or using arrays within the numpy library
list1 = [1,2,3,4,5]
# print('list1:', list1)
# list2=[[0],[1],[2],[3]]
# print('list2:', list2)
# list3 = [[0,1],[2,3]]
# print('list3:', list3)
# array1 = np.array([0,1,2,3])
# print('array1:', array1)
# array2 = np.array([[0],[1],[2],[3]])
# array3 = np.array([[0,1],[2,3]])
# print('array2:', array2)
# print('array3:', array3)
# =============================================================================

# =============================================================================
# #Arrays can also be defined using the numpy arange() functions which allows an array to
# #be prepopuated with values, either specifiyng a max value and it will count 1-max by 1, or giving 
# #lower and upper bounds and a step value, it ends before the upper bound. The linspace() from numpy does similar but it takes
# #a lower and upper bound and the number of steps to divide it.
# #print(np.arange(4), '\n', np.arange(0,2,0.5), '\n', np.linspace(0,1.5,4))
# =============================================================================


# =============================================================================
# array1 = np.array(list1)
# print('list1 :', list1[0],list1[4])
# 
# print('array1 :', array1[0],array1[4])
# 
# =============================================================================

#numpy can create arrays of zeros and ones of any size using np.ones() or np.zeros()


steps = 0.1
x = np.arange(-2,2+steps,steps)

y1 = x+2
y2 = x**2

plt.figure(figsize=(12,8))









