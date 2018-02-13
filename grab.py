from PIL import ImageGrab, ImageOps, Image
from helper_functions import img2vec
import os
import time
import numpy as np

def screen_grab(x, y, square_pad, image_size):
    box = (x, y, x + square_pad, y + square_pad)
    im = ImageGrab.grab(box)
#    im = resize(im, (80,80))
#    im.save("images/"+'full_snap__' +str(int(time.time())) + '.png', 'PNG')
    im = resize(im, image_size)
    im = img2vec(np.array(im), 0)
    return im    

def read_empty_grid():
    im = Image.open("images/empty_grid.png")
    im = img2vec(np.array(im), 0)
    return im

#e = read_empty_grid().T / 255.
#s = img2vec(np.array(Image.open("images/full_snap__1518535869.png")), 0).T / 255.
#print(np.abs(np.sum(np.subtract(e, s))))

def resize(image, size):
    return ImageOps.fit(image, size, Image.ANTIALIAS)
