from PIL import ImageGrab, ImageOps, Image
from helper_functions import img2vec
import os
import time
import numpy as np

def screen_grab(x, y, square_pad, image_size):
    box = (x, y, x + square_pad, y + square_pad)
    im = ImageGrab.grab(box)
    im = resize(im, image_size)
#    im.save(os.getcwd() + '\\full_snap__' +str(int(time.time())) + '.png', 'PNG')
    im = img2vec(np.array(im), 0)
    return im    

def resize(image, size):
    return ImageOps.fit(image, size, Image.ANTIALIAS)

