"""
Taken and modified from 'Viewing 3D volumetric data with matplotlib'
by Juan Nunez-Iglesias
https://www.datacamp.com/community/tutorials/matplotlib-3d-volumetric-data
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def ani3D(X, name):
    
        fig = plt.figure()
        # img = plt.imshow(X[2], animated=True)
        
        def updatefig(i):
            img.set_array(X[:,:,i])
            return img,
        
        ani = animation.FuncAnimation(fig, updatefig, frames=X.shape[2],
                                      interval=0.05, blit=True)
        # ani.save(name, fps=20, writer='PillowWriter')
        # plt.show(ani)