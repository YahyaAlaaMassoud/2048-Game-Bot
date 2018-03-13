from PIL import ImageGrab, ImageOps, Image
from helper_functions import img2vec
import os
import time
import numpy as np
from base64 import b16encode

color_to_number = {
                     (205,193,180): 0,
                     (238,228,218): 2,
                     (237,224,200): 4,
                     (242,177,121): 8,
                     (245,149,99): 16,
                     (246,124,95): 32,
                     (246,94,59): 64,
                     (237,207,114): 128,
                     (237,204,97): 256,
                     (237,200,80): 512,
                     (237,197,63): 1024,
                     (237,194,46): 2048
                  }
#    example:
#    save_image(image, "images/full_snap_" + str(int(time.time())) + ".png", "PNG")
def save_image(image, path, type_format = "PNG"):
    image.save(path, type_format)

def screen_grab(x, y, square_pad):
    box = (x, y, x + square_pad, y + square_pad)
    im = ImageGrab.grab(box)
    return im

def get_matrix(image):
    image = image.convert('RGB')
    matrix = []
    keys = [key for key in color_to_number.keys()]
    x = 30
    max_value = 0
    for i in range(4):
        y = 30
        for j in range(4):
            r, g, b = image.getpixel((x, y))
            ans = (0, 0, 0)
            diff = 10000
            ok = False
            for key in keys:
                cur_diff = abs(r - key[0]) + abs(g - key[1]) + abs(b - key[2])
                if cur_diff < diff:
                    ans = key
                    diff = cur_diff
                if key == (r, g, b):
                    ok = True
            matrix.append(color_to_number[ans])
            max_value = max(max_value, int(color_to_number[ans]))
#            if ok == False:
#                print(r,g,b)
#                print(ans)
#                print()
            y = y + 120
        x = x + 120
    return matrix, max_value

def get_state(x, y, square_pad, image_size):
    im = screen_grab(x, y, square_pad)
    matrix_full, max_value = get_matrix(im)
    matrix = normalize_matrix(matrix_full)
    return np.array(matrix).reshape(1, len(matrix)), np.array(matrix_full).reshape(1, len(matrix_full)), max_value

def normalize_matrix(mat):
    maxi = max(mat)
    normalized = [float(max(0, np.log2(x) / np.log2(maxi))) for x in mat]
    return normalized

def read_image(path):
    im = Image.open(path)
    im = img2vec(np.array(im), 0)
    return im

def resize(image, size):
    return ImageOps.fit(image, size, Image.ANTIALIAS)