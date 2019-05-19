#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 08:12:59 2019

@author: jing
"""

import numpy as np

import matplotlib.pyplot as plt
 

FE=np.loadtxt('loss_FE.txt',dtype=np.float32)

RK2=np.loadtxt('loss_2RK.txt',dtype=np.float32)

RK4=np.loadtxt('loss_4RK.txt',dtype=np.float32)

Ad=np.loadtxt('loss_Adams.txt',dtype=np.float32)

Lp=np.loadtxt('loss_Leap.txt',dtype=np.float32)

BE=np.loadtxt('loss_BE.txt',dtype=np.float32)

Mol=np.loadtxt('loss_Mol.txt',dtype=np.float32)

CN=np.loadtxt('loss_CN.txt',dtype=np.float32)

#x = range(FE.shape[1]-3) 
x = range(3,125)
plt.figure()
plt.ylim(17.5, 19)
p1, = plt.plot(x,FE[1,3:125], 'g--')
p2, =plt.plot(x,RK2[1,3:125], 'r--')
p3, =plt.plot(x,RK4[1,3:125], 'b')
p4, =plt.plot(x,Ad[1,3:125], 'y--')
p5, =plt.plot(x,BE[1,3:125], 'r-')
p6, = plt.plot(x,Mol[1,3:125], 'y-')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend([p1, p2,p3, p4,p5, p6], ["Forward Euler", "2nd RK","4th RK","Adams Bashforth","Backward Euler","Adams Moulton",], loc='upper right')
plt.title("Comparision among different methods")

plt.savefig("plot1.jpg")
plt.figure()
p7, = plt.plot(x,Lp[1,3:125], 'b')
p8,= plt.plot(x,CN[1,3:125], 'r')
plt.legend([p7, p8], ["Leap frog", "Crank-Nicolson"], loc='upper right')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("Comparision between Leapfrog and Crank-Nicolson")
plt.savefig("plot2.jpg")