#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 22:48:41 2025

@author: diller
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

def genNums(n):
    randGen = np.random.default_rng()
    randVector = randGen.random(n)
    randVector = randVector * 2 * np.pi
    xVector = np.cos(randVector)
    yVector = np.sin(randVector)
    cordVector = np.stack((xVector, yVector))
    return cordVector

n = 1000  # Example number of points
cordVector = genNums(n)
x_coords = cordVector[0, :]
y_coords = cordVector[1, :]

# Create a pandas DataFrame
df = pd.DataFrame({'x': x_coords, 'y': y_coords})

# Define the bin edges
x_bins = np.linspace(df['x'].min(), df['x'].max(), 20)  # Adjust bin number as needed
y_bins = np.linspace(df['y'].min(), df['y'].max(), 20)  # Adjust bin number as needed

# Bin the data
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
ax.plot_surface(x_mesh, y_mesh, bin_counts.values, cmap='viridis')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Frequency')
ax.set_title('3D Histogram of Binned Data')

plt.show()