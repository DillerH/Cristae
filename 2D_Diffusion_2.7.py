#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 11:38:28 2025

@author: diller
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi as pi
import pandas as pd
import time as time
from matplotlib.animation import FFMpegWriter


fontTit = {'family':'Arial','size':'18'}
fontSub ={'family':'Arial','size':'18'}


stepCount = 20000000 #in ns
printData = True
tau = 10**(-9) #determines 'time scale' i.e. ms, µs, ns
frameSizeMax = 200 #nm
frameSizeMin = 0 #nm

class Particle:
    def __init__(self, redox, leaf, position = [0,0]):
        self.position = [0, 0]  
        self.redox = redox
        self.leaf = leaf
        self.bound = False
        self.Dict = 0
        self.oob = False
    def reduce(self):
        self.redox = self.redox + 1
    def oxidize(self):
        self.redox = self.redox - 1

class ParticleGroup:
    def __init__(self, particleList, r, D, label):
        self.numParticles = len(particleList)
        self.particles = np.array(particleList, dtype=object) # Store as object array
        self.positions =  np.array([[particle.position[0], particle.position[1]] for particle in particleList]) #check this line in the future from chatGPT
        self.D = D #nm^2/s
        self.radius = r
        self.label = label
        self.frameNum = 0
        self.oob = np.zeros((len(particleList)))
        self.redox = np.zeros((len(particleList)))
        self.lossRate = [0, 0, 0] #[currentRate, number of particles, time]
        self.delta = np.sqrt(4 * tau * D)   #nm step length in nm for 1µs
        
    def __len__(self):
        return self.numParticles

    def spawnRandParticle(self):
        
        #for a given tick
        if self.lossRate[0] <= 1: #important for if loss rate is >1
            chanceAdd = self.lossRate[0]
        else:
            chanceAdd = 1
        chancePass = 1 - chanceAdd
        #print('{} + {} = {}'.format(chanceAdd,chancePass,chanceAdd+chancePass))
        if np.random.choice([True,False],p=[chanceAdd, chancePass]) == True:
            randGen = np.random.default_rng()
            randX = randGen.random() * frameSizeMax
            randY = randGen.random() * frameSizeMax
            randCord = np.array((randX, randY))

        
            if self.label == 'Q10':
                QRat = Qratio(self)
                if np.random.choice([True, False],p=[QRat, 1-QRat])== True:
                    self.positions = np.vstack((self.positions, randCord))
                    QH2 = Particle(2,0, position = randCord)
                    self.particles = np.append(self.particles,QH2)
                    self.redox = np.append(self.redox, 2)
                    self.oob = np.append(self.oob, 0)
                    self.numParticles += 1
                    #print('Added Particle')
                else:
                    self.positions = np.vstack((self.positions, randCord))
                    Q = Particle(0,0, position = randCord)
                    self.particles = np.append(self.particles, Q)
                    self.redox = np.append(self.redox, 0)
                    self.oob = np.append(self.oob, 0)
                    self.numParticles += 1
            else: 
                self.positions = np.vstack((self.positions, randCord))
                Q = Particle(0,0, position = randCord)
                self.particles = np.append(self.particles,Particle(2, 0))
                self.redox = np.append(self.redox, 2)
                self.oob = np.append(self.oob, 0)
                self.numParticles += 1
    
    def checkBoxEdge(self, frameSize):
        if self.numParticles > 0:
            for k in range(0,self.numParticles):
                x = self.particles[k].position[0]
                y = self.particles[k].position[1]
                if x > frameSize  or y > frameSize or x < frameSizeMin or y < frameSizeMin:
                    self.particles[k].oob = True
            updateBounds(self)
    def GPTremoveOOB(self, t):
    # A list to store indices of particles to be removed
        particles_to_remove = []

    # Loop over all particles and check if they are out of bounds
        for k, particle in enumerate(self.particles):
            if particle.oob:  # Only consider particles that are out of bounds
            # Calculate the distribution and removal probabilities for each boundary condition
                if particle.position[0] >= frameSizeMax and frameSizeMin <= particle.position[1] <= frameSizeMax:
                    pdfx, pdfy = pdf(self.D, self.delta, 1, frameSizeMax)
                    Gauss = np.vstack((pdfx, pdfy))
                    IntoBox = np.where(Gauss[0] <= frameSizeMax, Gauss, np.nan)
                    ooBox = np.where(Gauss[0] >= frameSizeMax, Gauss, np.nan)
                    
                elif particle.position[0] <= frameSizeMin and frameSizeMin <= particle.position[1] <= frameSizeMax:
                    pdfx, pdfy = pdf(self.D, self.delta, 1, frameSizeMin)
                    Gauss = np.vstack((pdfx, pdfy))
                    IntoBox = np.where(Gauss[0] <= frameSizeMin, Gauss, np.nan)
                    ooBox = np.where(Gauss[0] >= frameSizeMin, Gauss, np.nan)
                        
                elif particle.position[1] >= frameSizeMax and frameSizeMin <= particle.position[0] <= frameSizeMax:
                    pdfx, pdfy = pdf(self.D, self.delta, 1, frameSizeMax)
                    Gauss = np.vstack((pdfx, pdfy))
                    IntoBox = np.where(Gauss[0] <= frameSizeMax, Gauss, np.nan)
                    ooBox = np.where(Gauss[0] >= frameSizeMax, Gauss, np.nan)

                elif particle.position[1] <= frameSizeMin and frameSizeMin <= particle.position[0] <= frameSizeMax:
                    pdfx, pdfy = pdf(self.D, self.delta, 1, frameSizeMin)
                    Gauss = np.vstack((pdfx, pdfy))
                    IntoBox = np.where(Gauss[0] <= frameSizeMin, Gauss, np.nan)
                    ooBox = np.where(Gauss[0] >= frameSizeMin, Gauss, np.nan)

            # Compute chances
                chanceIn = np.nansum(IntoBox[1]) / (np.nansum(IntoBox[1]) + np.nansum(ooBox[1]))
                chanceOut = np.nansum(ooBox[1]) / (np.nansum(IntoBox[1]) + np.nansum(ooBox[1]))
            
            # Use np.random.choice to decide whether to keep or remove the particle
                if np.random.choice([True, False], p=[chanceIn, chanceOut]) == False:
                    particles_to_remove.append(k)  # Store index of particle to be removed

    # Now remove particles in reverse order to avoid index shifting
        for index in reversed(particles_to_remove):
            self.particles = np.delete(self.particles, index, axis=0)
            self.positions = np.delete(self.positions, index, axis=0)
            self.numParticles -= 1
            self.lossRate[1] += 1  # Increase number of lost particles
        
        # Update lossRate[2] (time) and lossRate[0] (loss rate)
            self.lossRate[2] = t if t != 0 else 1
            self.lossRate[0] = self.lossRate[1] / self.lossRate[2]

    def removeOOB(self, t): #fix gauss incorrect if statement
        
        boolPrint = False
        indexList, oobParticles = [], []
        for k, particle in enumerate(self.particles):
            if particle.oob == True:
                indexList.append(k)
                oobParticles.append(particle)
        if len(oobParticles) == 0: # consider nesting in one for loop here instead of two
            return
            # x > 200     
        #print(len(oobParticles))
        for k, particle in enumerate(oobParticles):
            if particle.position[0] >= frameSizeMax and particle.position[1] <= frameSizeMax and particle.position[1] >= frameSizeMin:
                pdfx, pdfy = pdf(self.D, self.delta, 1, frameSizeMax)
                Gauss = np.vstack((pdfx,pdfy))
            #print(particle.position[1])
                IntoBox = np.where(Gauss[0] <= frameSizeMax, Gauss, np.nan) #querys pdfx and replaces
                ooBox = np.where(Gauss[0] >= frameSizeMax, Gauss, np.nan)
                #check distribution
                if boolPrint:
                    plt.plot(IntoBox[0], IntoBox[1],color='green')
                    plt.plot(ooBox[0], ooBox[1],color='red')
                    plt.xlim(-(frameSizeMin+(2*frameSizeMax)),(2*frameSizeMax))
                    plt.show()
                    
                chanceIn = np.nansum(IntoBox[1]) /(np.nansum(IntoBox[1]) + np.nansum(ooBox[1]))
                chanceOut = np.nansum(ooBox[1]) /(np.nansum(IntoBox[1]) + np.nansum(ooBox[1]))
                keepBool = np.random.choice([True, False], p=[chanceIn,chanceOut])
                if keepBool == False:
                    #print('Chunk 1 k={}, len={}'.format(k,len(self)))
                    #[currentRate, number of particles, time]
                    self.particles = np.delete(self.particles,indexList[k] , axis = 0)
                    self.positions = np.delete(self.positions, indexList[k], 0)
                    self.numParticles -= 1
                    self.lossRate[1] += 1 
                    self.lossRate[2] = t if t != 0 else 1
                    self.lossRate[0] = self.lossRate[1] / self.lossRate[2]
                    
                    for j in range(k + 1, len(indexList)):
                        indexList[j] -= 1
                    
        #x< 0
        
        

            if particle.position[0] <= frameSizeMin and particle.position[1] <= frameSizeMax and particle.position[1] >= frameSizeMin:
                pdfx, pdfy = pdf(self.D, self.delta, 1, frameSizeMin)
                Gauss = np.vstack((pdfx,pdfy))
                IntoBox = np.where(Gauss[0] <= frameSizeMin, Gauss, np.nan) #querys pdfx and replaces
                ooBox = np.where(Gauss[0] >= frameSizeMin, Gauss, np.nan)
                #check distribution
                if False:
                    plt.plot(IntoBox[0], IntoBox[1],color='green')
                    plt.plot(ooBox[0], ooBox[1],color='red')
                    plt.xlim(-(frameSizeMin+(2*frameSizeMin)),(2*frameSizeMin))
                    plt.show()
                    
                chanceIn = np.nansum(IntoBox[1]) /(np.nansum(IntoBox[1]) + np.nansum(ooBox[1]))
                chanceOut = np.nansum(ooBox[1]) /(np.nansum(IntoBox[1]) + np.nansum(ooBox[1]))
                keepBool = np.random.choice([True, False], p=[chanceIn,chanceOut])
                
                if keepBool == False:
                   # print('Chunk 2 k={}, len={}'.format(k,len(self)))
                    #print('Removing particle {}'.format(k))
                    self.particles = np.delete(self.particles,indexList[k] , axis = 0)
                    self.positions = np.delete(self.positions, indexList[k], 0)
                    self.numParticles -= 1
                    self.lossRate[1] += 1
                    if t != 0:
                        self.lossRate[2] = t 
                    else:
                        self.lossRate[2] = 1
                    self.lossRate[0] = self.lossRate[1] / self.lossRate[2]
                    for j in range(k + 1, len(indexList)):
                        indexList[j] -= 1
            # if y > 200
            if particle.position[1] >= frameSizeMax and particle.position[0] <= frameSizeMax and particle.position[0] >= frameSizeMin:
                pdfx, pdfy = pdf(self.D, self.delta, 1, frameSizeMax)
                Gauss = np.vstack((pdfx,pdfy))
        
            #print(particle.position[1])
                IntoBox = np.where(Gauss[0] <= frameSizeMax, Gauss, np.nan) #querys pdfx and replaces
                ooBox = np.where(Gauss[0] >= frameSizeMax, Gauss, np.nan)
                #check distribution
                if False:
                    plt.plot(IntoBox[0], IntoBox[1],color='green')
                    plt.plot(ooBox[0], ooBox[1],color='red')
                    plt.xlim(-(frameSizeMin+(2*frameSizeMax)),(2*frameSizeMax))
                    plt.show()
                    
                chanceIn = np.nansum(IntoBox[1]) /(np.nansum(IntoBox[1]) + np.nansum(ooBox[1]))
                chanceOut = np.nansum(ooBox[1]) /(np.nansum(IntoBox[1]) + np.nansum(ooBox[1]))
                keepBool = np.random.choice([True, False], p=[chanceIn,chanceOut])
                if keepBool == False:
                    #print('Chunk 3 k={}, len={}'.format(k,len(self)))
                    #print('Removing particle {}'.format(k))
                    self.particles = np.delete(self.particles,indexList[k] , axis = 0)
                    self.positions = np.delete(self.positions, indexList[k], 0)
                    self.numParticles -= 1
                    self.lossRate[1] += 1 
                    if t != 0:
                        self.lossRate[2] = t 
                    else:
                        self.lossRate[2] = 1
                    self.lossRate[0] = self.lossRate[1] / self.lossRate[2]
                    
                    for j in range(k + 1, len(indexList)):
                        indexList[j] -= 1
                #y < 0
            if particle.position[1] <= frameSizeMin and particle.position[0] <= frameSizeMax and particle.position[0] >= frameSizeMin:
                pdfx, pdfy = pdf(self.D, self.delta, 1, frameSizeMin)
                Gauss = np.vstack((pdfx,pdfy))
        
            #print(particle.position[1])
                IntoBox = np.where(Gauss[0] <= frameSizeMin, Gauss, np.nan) #querys pdfx and replaces
                ooBox = np.where(Gauss[0] >= frameSizeMin, Gauss, np.nan)
                #check distribution
                if False:
                    plt.plot(IntoBox[0], IntoBox[1],color='green')
                    plt.plot(ooBox[0], ooBox[1],color='red')
                    plt.xlim(-(frameSizeMin+(2*frameSizeMin)),(2*frameSizeMin))
                    plt.show()
                    
                chanceIn = np.nansum(IntoBox[1]) /(np.nansum(IntoBox[1]) + np.nansum(ooBox[1]))
                chanceOut = np.nansum(ooBox[1]) /(np.nansum(IntoBox[1]) + np.nansum(ooBox[1]))
                keepBool = np.random.choice([True, False], p=[chanceIn,chanceOut])
                
                if keepBool == False:
                    #print('Chunk 4 k={}, len={}'.format(k,len(self)))
                    #print('Removing particle {}'.format(k))
                    self.particles = np.delete(self.particles,indexList[k] , axis = 0)
                    self.positions = np.delete(self.positions, indexList[k], 0)
                    self.numParticles -= 1
                    self.lossRate[1] += 1
                    if t != 0:
                        self.lossRate[2] = t 
                    else:
                        self.lossRate[2] = 1
                            
                    self.lossRate[0] = self.lossRate[1] / self.lossRate[2]
                    for j in range(k + 1, len(indexList)):
                        indexList[j] -= 1

        
    def randStep(self, time = 0):
        self.frameNum += 1
        randGen = np.random.default_rng()#calls random generator
        randVector = randGen.random((self.numParticles)) * 2 * pi # gens 0-1 array length of number of particles % & scales to radians
        xVector = np.cos(randVector) * self.delta
        yVector = np.sin(randVector) * self.delta
        cordVector = np.stack((xVector, yVector), axis=1)
        self.positions = self.positions + cordVector
        updateParticles(self)      
            
    def initRandFrame(self, frameSize):
        randGen = np.random.default_rng()
        randXVector = randGen.random(self.numParticles) * frameSize
        randYVector = randGen.random(self.numParticles) * frameSize
        cordVector = np.stack((randXVector, randYVector), axis=1)
        
        self.positions = cordVector
        #update particle positions
        for i, particle in enumerate(self.particles):

            particle.position[0] = self.positions[i, 0] #x
            particle.position[1] = self.positions[i, 1] #y
            
    def reductionActivity(self, substrateGroup, reactionRadius):
        
        for i, protein in enumerate(self.particles):
            #setting bounds
            for k, substrate in enumerate(substrateGroup.particles):
                distance = np.sqrt((protein.position[0] - substrate.position[0])**2 + (protein.position[1] - substrate.position[1])**2)

                if distance <= reactionRadius and substrate.redox == 0 :
                    substrate.redox = substrate.redox + 2
                    
    def oxidationActivity(self, substrateGroup, reactionRadius):
        
        
        
        for i, protein in enumerate(self.particles):
            #setting bounds
            for k, substrate in enumerate(substrateGroup.particles):
                distance = np.sqrt((protein.position[0] - substrate.position[0])**2 + (protein.position[1] - substrate.position[1])**2)

                if distance <= reactionRadius and substrate.redox == 2 :
                    for k, sub in enumerate(substrateGroup.particles):
                        dist = np.sqrt((protein.position[0] + sub.position[0])**2 + (protein.position[1] + sub.position[1])**2)
                        
                    if dist <= reactionRadius and sub.redox==0:
                        substrate.redox -= 2
                        
def updateBounds(particles):
    for i, particle in enumerate(particles.particles):
        particles.oob[i] = particle.oob

def pdf(D, delta, t, mu):
    delta = np.sqrt(4 * D * tau)
    sigma = np.sqrt(2 * D * t)
    x = np.linspace(-6*sigma - mu, 6*sigma + mu, num=100000)
    
    term1 = 2*delta / np.sqrt(4 * pi * t * D)
    term2 = np.exp((-(x-mu)**2)/(4 * t * D))
    y = term1 * term2 
    
    return x, y

def updateParticles(particles):
    #updating particles
    for i, particle in enumerate(particles.particles):
        #update particle positions
        particle.position[0] = particles.positions[i, 0] #x
        particle.position[1] = particles.positions[i, 1] #y
    
def plotFrame(substrateGroup, proteinGroupI, proteinGroupIII, time):
     
    #print('Q1\n',substrateGroup.positions)
    
    #print('Protein\n',proteinGroupI.positions)
    Q0eList, Q1eList, Q2eList = [], [], []
    Q0ePresent, Q1ePresent, Q2ePresent = False, False, False

    for Q in substrateGroup.particles:
        #print(Q.redox)
        if Q.redox == 0:
            Q0eList.append(Q)
        if Q.redox == 1:
            Q1eList.append(Q)
        if Q.redox ==2:
            Q2eList.append(Q)
    
    #print(Q0eList, Q1eList, Q2eList)
    if len(Q0eList) > 0:
        Q0e = ParticleGroup(Q0eList, 1, 1, 'Q10.e0')
        Q0ePresent = True
    if len(Q1eList) > 0:
        Q1e = ParticleGroup(Q1eList, 1, 1, 'Q10.e1')
        Q1ePresent = True
    if len(Q2eList) > 0:
        Q2e = ParticleGroup(Q2eList, 1, 1, 'Q10.e1')
        Q2ePresent = True

    
    fig, ax = plt.subplots()
    #print(Q0e.positions)
    ax.scatter(proteinGroupI.positions[:,0], proteinGroupI.positions[:,1],s=200, label = 'CI', color = 'crimson')
    ax.scatter(proteinGroupIII.positions[:,0], proteinGroupIII.positions[:,1],s=80, label = 'CIII', color = 'slateblue')
    if Q0ePresent:
        ax.scatter(Q0e.positions[:,0], Q0e.positions[:,1], label = 'CoQ',s=20, color = 'lightgray')
    if Q1ePresent:
        ax.scatter(Q1e.positions[:,0], Q1e.positions[:,1], label = 'SQ',s=20, color = 'dimgray')
    if Q2ePresent:
        ax.scatter(Q2e.positions[:,0], Q2e.positions[:,1], label = r'QH$_2$',s=20, color = 'black') 
    ax.set_xlabel('x (nm)',fontdict=fontSub)
    ax.set_ylabel('y (nm)',fontdict=fontSub)
    ax.set_title('Reduction Activity of Simulated Particles after '+str(time/1000)+' µs',fontdict=fontTit)
    ax.legend()
    plt.savefig('FrameShot1.png',dpi=600)
    plt.show()
            
def plot(particles):
    hist, xedges, yedges = np.histogram2d(particles.positions[:, 0], particles.positions[:, 1], bins=20)
    plt.imshow(hist.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto', cmap='terrain')
    plt.colorbar(label='Counts')
    plt.xlabel('x (nm)')
    plt.ylabel('y (nm)')
    plt.title('2D Histogram of Particle Positions '+str((stepCount)/1000)+' milliseconds/s')
    plt.show()

    # 3D representation of a 2D histo
    df = pd.DataFrame(particles.positions, columns=['x', 'y'])

    # Reusing bin edges from 2D histogram
    x_bins = xedges
    y_bins = yedges

    df['x_bin'] = pd.cut(df['x'], bins=x_bins)
    df['y_bin'] = pd.cut(df['y'], bins=y_bins)

    # Calculate the frequency in each bin
    bin_counts = df.groupby(['x_bin', 'y_bin']).size().unstack(fill_value=0)

    # Get the bin centers for plotting
    x_bin_centers = (x_bins[:-1] + x_bins[1:]) / 2
    y_bin_centers = (y_bins[:-1] + y_bins[1:]) / 2

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create meshgrid of bin centers
    x_mesh, y_mesh = np.meshgrid(x_bin_centers, y_bin_centers)

    # Plot the 3D surface
    ax.plot_surface(x_mesh, y_mesh, bin_counts.values, cmap='terrain')
    
    # Set labels and title
    ax.set_xlabel('x (nm)')
    ax.set_ylabel('y (nm)')
    ax.set_zlabel('Frequency')
    ax.set_title('3D Representation of the 2D Histogram '+str((stepCount)/1000)+' millisecond/s')
    plt.show()
    return

def plotTrace(tracex, tracey):
   # print('Starting Trace plot')
    zipList = zip(tracex, tracey)
    df = pd.DataFrame(zipList,columns=['x', 'y'])

    
    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(df)+1)))
    
    for i in range(0, len(df)-1 ):
        c = next(color)
        plt.plot(df.x.iloc[i:i+2], df.y.iloc[i:i+2],color = c)
        plt.xlabel('x(nm)')
        plt.ylabel('y(nm)')
        plt.title('Single Molecule Trace '+str((stepCount)/1000)+' millisecond/s')
        plt.grid()
        
    #plt.xlim(-250,250)
    #plt.ylim(-250,250)
    plt.show()
       
def initialConditions(allGroups, frameSize): #nm
    for group in allGroups:
        group.initRandFrame(frameSize)

def Qratio(QuinoneGroup):
    Qredoxes = []
    if len(QuinoneGroup)==0:
        return 0
    for quinone in QuinoneGroup.particles:
        Qredoxes.append(quinone.redox)
        
    unique, counts = np.unique(Qredoxes, return_counts=True)
    QDict = dict(zip(unique, counts))
    QDict.setdefault(0, 0)
    QDict.setdefault(2, 0)
    #print(QDict[0])

    return QDict[2]/len(QuinoneGroup)
  
def plotQPool(Ratio):

    x = [i/1000 for i in range(0,len(Ratio))]
    y = Ratio
    fig, ax = plt.subplots(figsize=(4.4,4.4))
    ax.plot(x,y,label=r'$\frac{QH_2}{Q_{tot}}$')
    ax.set_xlabel(r'Time $(\mu s)$',fontdict=fontSub)
    ax.set_ylim(0,1)
    ax.set_xlim(0,(stepCount/1000))
    ax.set_ylabel(r'$\frac{QH_2}{Q_{tot}}$', fontdict=fontSub)
    ax.set_title('Ratio of Reduced Quinone \n as a Function of Time',fontdict=fontTit)
    plt.savefig('QPool1.png',dpi=600)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plotQNums(t, y):
    x = [i/1000 for i in range(0,t)]
    fig, ax = plt.subplots()
    ax.plot(x,y)
    ax.set_xlabel('Time (µs)',fontdict=fontSub)
    ax.set_ylabel('Number of Quinones in Pool',fontdict=fontSub)
    ax.set_title('Variation in Quinone', fontdict=fontTit)
    plt.savefig('QVar1.png',dpi=600)
    plt.show()

def plotMovie():

    frameCI = plt.scatter([], [], label='CI', color ='crimson')
    frameCIII = plt.scatter([], [], label='CIII', color = 'slateblue')
    plt.xlabel('x (nm)',fontdict=fontSub)
    plt.ylabel('y (nm)',fontdict=fontSub)
    plt.title('Reduction Activity of Simulated Particles',fontdict=fontTit)
    #plt.text(1,1, 't = '+str(t)+' ns')
    plt.xlim(0, 200)
    plt.ylim(0, 200)
    plt.legend(loc='upper right')
    frameQH2 = plt.scatter([], [], label ='QH2', color ='black')
    frameQ = plt.scatter([], [], label='Q', color='lightgray')
    """
    #ax.scatter(proteinGroupI.positions[:,0], proteinGroupI.positions[:,1],s=200, label = 'CI', color = 'crimson')
    #ax.scatter(proteinGroupIII.positions[:,0], proteinGroupIII.positions[:,1],s=80, label = 'CIII', color = 'slateblue')
    if Q0ePresent:
        ax.scatter(Q0e.positions[:,0], Q0e.positions[:,1], label = 'CoQ',s=20, color = 'lightgray')
    if Q1ePresent:
        ax.scatter(Q1e.positions[:,0], Q1e.positions[:,1], label = 'SQ',s=20, color = 'dimgray')
    if Q2ePresent:
        ax.scatter(Q2e.positions[:,0], Q2e.positions[:,1], label = r'QH$_2$',s=20, color = 'black') 
    #plt.savefig('FrameShot1.png',dpi=600)
    """
    return frameCI, frameCIII, frameQH2, frameQ
 
def makeMovie():
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata)
    
    
    
    Qlist = list(Particle(0, 1) for i in range(0,100))
    ComplexIlist = list(Particle(1, 1) for i in range(0,2)) 
    ComplexIIIlist = list(Particle(1, 1) for i in range(0,8)) 
    
    Q =ParticleGroup(Qlist, 0.5, 8.2*10**6, 'Q10')
    ComplexI = ParticleGroup(ComplexIlist, 0.5, 6.72*10**5, 'CI')
    ComplexIII = ParticleGroup(ComplexIIIlist, 0.5, 1.0*10**6, 'CIII')
    
    Q.initRandFrame(200)
    ComplexI.initRandFrame(200)
    ComplexIII.initRandFrame(200)
    
    fig = plt.figure()
    
    CIframe, = plt.plot([], [], 'o',label='CI', color ='crimson', markersize=20)
    CIIIframe, = plt.plot([], [],'o', label='CIII', color = 'slateblue',markersize=8)
    plt.xlabel('x (nm)',fontdict=fontSub)
    plt.ylabel('y (nm)',fontdict=fontSub)
    plt.title('Reduction Activity of Simulated Particles',fontdict=fontTit)
    #plt.text(1,1, 't = '+str(t)+' ns')
    plt.xlim(0, 200)
    plt.ylim(0, 200)
    QH2frame, = plt.plot([], [],'o', label ='QH2', color ='black',markersize=5)
    Qframe, = plt.plot([], [],'o', label='CoQ', color='lightgray',markersize=5)
    plt.legend(loc='upper right')
    
    with writer.saving(fig, 'Diffusion.mp4',100):
        for i in range(0,1000):
            Q0eList, Q2eList = [], []
            Q0ePresent, Q2ePresent = False,  False
            if i != 0:
                for text in plt.gca().texts:
                    text.set_visible(False)
            plt.text(0,0,'time = '+str(i)+' ns')
            for q in Q.particles:
                #print(Q.redox)
                if q.redox == 0:
                    Q0eList.append(q)
                if q.redox == 2:
                    Q2eList.append(q)
            #print('complete')
            #print(Q0eList, Q1eList, Q2eList)
            if len(Q0eList) > 0:
                Q0e = ParticleGroup(Q0eList, 1, 1, 'Q')
                Q0ePresent = True
            if len(Q2eList) > 0:
                Q2e = ParticleGroup(Q2eList, 1, 1, 'QH2')
                Q2ePresent = True
            CIframe.set_data(ComplexI.positions[:,0], ComplexI.positions[:,1])
            CIIIframe.set_data(ComplexIII.positions[:,0], ComplexIII.positions[:,1])
            if Q0ePresent:
                Qframe.set_data(Q0e.positions[:,0], Q0e.positions[:,1])
            if Q2ePresent:
                QH2frame.set_data(Q2e.positions[:,0], Q2e.positions[:,1])
            writer.grab_frame()
            
            Q.randStep()
            ComplexI.randStep()
            ComplexIII.randStep()
            
            ComplexI.reductionActivity(Q, 4)
            ComplexIII.oxidationActivity(Q, 4)
            
            #ax.scatter(Q0e.positions[:,0], Q0e.positions[:,1], label = 'CoQ',s=20, color = 'lightgray')
    
    
def main():
    
    Q10list = list(Particle(0, 1) for i in range(0,50))
    CIlist = list(Particle(1, 1) for i in range(0,1)) 
    CIIIlist = list(Particle(1, 1) for i in range(0,4)) 
    
    Q10 = ParticleGroup(Q10list, 0.5, 8.2*10**6, 'Q10')
    #print(Q10.delta)
    CI = ParticleGroup(CIlist, 0.5, 6.72*10**5, 'CI')

    CIII = ParticleGroup(CIIIlist, 0.5, 1.0*10**6, 'CIII')
    print('Q10 δ = {} nm\nCI δ = {} nm\nCIII δ = {} nm'.format(Q10.delta,CI.delta,CIII.delta))
    
    allMolecules = [Q10, CI, CIII]

    #CI = ParticleGroup(POPClist, 0.68, 7.79*10**6)
    
    traceListx = []
    traceListy = []
    QRatios = []
    QNums = []
    
    #df_Q10 = pd.DataFrame(Q10.positions, columns=['x0','y0'])
    #df_printout.index.name = 'Q10'
    
    
    initialConditions(allMolecules, frameSizeMax)
    plotFrame(Q10, CI, CIII, 0)
    for i in range(stepCount):
        #print(Q10.lossRate)
        particle1 = Q10.particles[0]
        traceListx.append(particle1.position[0])
        traceListy.append(particle1.position[1])
        if len(Q10) <= 75:
            Q10.spawnRandParticle()
        #if i % 400 == 0 and i!=0 and len(CI) <= 1 + len(CIlist):
            #CI.spawnRandParticle(tau, frameSizeMax)
        #if i % 100 == 0 and i!=0 and len(CI) <= 2 + len(CIIIlist):
            #CIII.spawnRandParticle(tau, frameSizeMax)
        Q10.randStep()
        #CI.randStep()
        #CIII.randStep()
        #print(Q10.delta)
        Q10.checkBoxEdge(frameSizeMax)
        #CI.checkBoxEdge(frameSizeMax)
        #CIII.checkBoxEdge(frameSizeMax)
        Q10.removeOOB(i)
        #print(Q10.lossRate)
        CI.reductionActivity(Q10, 4)
        CIII.oxidationActivity(Q10, 4)
        

        QRatios.append(Qratio(Q10))
        QNums.append(len(Q10))
        #print(QRatios)
            #CI.spawnRandParticle(tau, 200)
            #CIII.spawnRandParticle(tau, 200)
        #print(len(Q10))
        #print(len(Q10))
        

        if printData == True and i % 10000 == 0 :
            #df_tempQ10 = pd.DataFrame(Q10.positions,columns=['x'+str(i),'y'+str(i)])
            #df_Q10 = pd.concat([df_Q10, df_tempQ10], axis=1)
            print(i)
            #print(len(Q10))
    
    #print(len(Q10))
    #plot(Q10)
    #print(len(Q10))
    #print(QRatios)
    plotQPool(QRatios)
    plotQNums(stepCount, QNums)
    plotFrame(Q10, CI, CIII, stepCount)


if __name__ == '__main__':
    s = time.time()
    #main()
    makeMovie()
    e = time.time()
    print(e-s)