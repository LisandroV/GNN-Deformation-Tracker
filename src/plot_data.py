"""This file plots the training dataset as a level set."""

import os
import numpy as np

import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

import data.train2_data as data

def plot_data(level_set, finger_positions):
    #poligon_indexes.append(poligon_indexes[0])

    title = "Level Set"
    fig = plt.figure(title)
    fig.suptitle(title)
    ax = fig.add_subplot(111, projection="3d")

    NCURVES = 100
    values = range(NCURVES)
    jet = plt.get_cmap("nipy_spectral")
    cNorm = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)

    # plots every control point history separately
    for i in values:
        colorVal = scalarMap.to_rgba(values[i])
        polygon = level_set[i,:,:]
        t = np.array([i]*47)
        ax.plot(t, polygon[:, 0], polygon[:, 1], color=colorVal, alpha=0.5)

    # Plot finger position
    ax.plot(np.array(values), finger_positions[:, 0], finger_positions[:, 1], color='blue', alpha=0.5)

    ax.set_xlabel("time")
    ax.set_ylabel("x")
    ax.set_zlabel("y")

    plt.show()


if __name__ == "__main__":

    # 100 steps in time
    # 47 points in the object contour
    # 2 coordinates (x,y)
    level_set = np.array(data.level_set) # shape: (100, 47, 2)
    finger_data = np.array(data.finger_data) # # shape: (100, 4)

    plot_data(level_set, finger_data)
