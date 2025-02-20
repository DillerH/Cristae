#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 17:34:42 2025

@author: diller
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.stats import tstd
from scipy.constants import pi as pi


stepCount = 500000
particleCount = 200000
printstd = False
printDict = False
printIt = False
e = 2.71828

def genNums(n):
    vector = np.random.randint(low = 0, high = 2, size = n) #generates boolean array of 0 and 1
    vector = vector * 2                                     #multiplies each element in array by two, aresulting in 0,2 outputs
    vector = vector - 1                                     #subtracts one from each element resulting in -1 and 1                                           
    return vector

def iterateFull(vector, t, i):
    if printIt == True:
        for k in range(0, t):
            vector = vector + genNums(i) #initial input + random vector of -1 +1
            print(k) #print iteration for 'loading bar'
    else:
        for k in range(0, t):
            vector = vector + genNums(i) #initial input + random vector of -1 +1
        
        
    position, count = np.unique(vector, return_counts=True) #np. unique returns two lists of positions and their # of 
    countDict = dict(zip(position, count)) #zips and declares dictionary for easy bar chart latter and representation of the data
    
    count = count / i #Gives percentage
    relDict = dict(zip(position, count))#new dict for % in place of count
    
    
    #prints dictionaries and Standard deviation if requested
    if printstd == True:
        print('Std of Data',tstd(vector))
    if printDict == True:
        print('Positions w/ count: \n',countDict)
        print('Positions w/ Percent: \n', relDict)
    return countDict, relDict

def plotPositions(relativeDict, pdfx, pdfy):
    # uses matplotlib to make a figure
    fig, axes = plt.subplots(1,1)
    axes.set_title('Particle Distribution', font='Arial', fontsize = 22)
    bound = (findBound(stepCount))
    axes.bar(relativeDict.keys(), relativeDict.values(), width=2, label = 'Simulated Data', color = 'tab:brown')
    axes.plot(pdfx, pdfy, color='tab:green',linewidth=2, label= 'rms Displacement')
    axes.set_xlabel('Particle Position', font = 'Arial', fontsize = 16)
    axes.set_xbound(-bound,bound)
    plt.tick_params('both', labelsize = 12, size = 5, labelfontfamily='Arial')
    #axes.set_yticks(font='Arial', fontsize = 12)
    axes.set_ylabel('Relative Particle Abundance', font = 'Arial', fontsize = 16)
    axes.legend()
    #axes.set_yscale('logit')
    plt.show()
    
def findBound(t):
    return int(4*np.sqrt(t)) #returns 4sigma

        
def pdf(t, n):
    sigma = np.sqrt(t)
    print('Std of pdf', sigma)
    v1 = np.linspace(-4*sigma, 4*sigma, dtype=int)
    v2 = norm.pdf(v1, 0, sigma) *2 #why times 2 here????
    return v1, v2
def ownpdf(t):
    sigma = np.sqrt(t)
    v1 = np.linspace(-6*sigma, 6*sigma, dtype=int)
    term1 = 1 / np.sqrt(2 * pi * t)
    term2 = np.e**((-v1**2)/(2*t))
    ys = term1 * term2
    ys *= 2
    return v1, ys

def checkCount(Dict):
    print(sum(Dict.values()))
    return

def main():
    posVector = np.zeros(particleCount, dtype=int) #declares empty array
    posDict, relDict = iterateFull(posVector, stepCount, particleCount)
    if printDict == True:
        print('Relative Positions',relDict)
        print('True Positions',posDict)
    v1, v2 = ownpdf(stepCount)
    plotPositions(relDict, v1, v2)
    #checkCount(relDict)
    ownpdf(stepCount)
    
    
    

if __name__ == '__main__':
    main()