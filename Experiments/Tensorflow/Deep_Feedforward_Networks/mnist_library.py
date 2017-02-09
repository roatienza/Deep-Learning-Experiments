import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np

def readfile(path):
    try:
        data = img.imread(path)
        return data
    except:
        return np.array([])


def displayimage(path):
    data = img.imread(path)
    plt.imshow(data)
    plt.show()
    return