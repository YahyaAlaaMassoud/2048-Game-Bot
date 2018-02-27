from PIL import ImageGrab, ImageOps, Image
from helper_functions import img2vec
import os
import time
import numpy as np
from base64 import b16encode

#color_to_number = {
#                     "b'cdc1b4'": 0,
#                     "b'eee4da'": 2,
#                     "b'ede0c8'": 4,
#                     "b'f2b179'": 8,
#                     "b'f59563'": 16,
#                     "b'f67c5f'": 32,
#                     "b'f65e3b'": 64,
#                     "b'edcf72'": 128,
#                     "b'edcc61'": 256,
#                     "b'edc850'": 512,
#                     "b'edc53f'": 1024,
#                     "b'edc22e'": 2048
#                  }

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

def screen_grab(x, y, square_pad, image_size):
    box = (x, y, x + square_pad, y + square_pad)
    im = ImageGrab.grab(box)
#    im = resize(im, (80,80))
#    im.save("images/"+'full_snap__' +str(int(time.time())) + '.png', 'PNG')
#    print(im[1,1])
#    im = resize(im, image_size)
#    im = img2vec(np.array(im), 0)
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
    im = screen_grab(x, y, square_pad, image_size)
    matrix_full, max_value = get_matrix(im)
#    print(np.array(matrix).reshape(1, len(matrix)).reshape(4,4).T)
#    print()
    matrix = normalize_matrix(matrix_full)
    return np.array(matrix).reshape(1, len(matrix)), np.array(matrix_full).reshape(1, len(matrix_full)), max_value

def normalize_matrix(mat):
    maxi = max(mat)
    normalized = [float(max(0, np.log2(x) / np.log2(maxi))) for x in mat]
    return normalized

def read_empty_grid():
    im = Image.open("images/empty_grid.png")
    im = img2vec(np.array(im), 0)
    return im

#e = read_empty_grid().T / 255.
#s = img2vec(np.array(Image.open("images/full_snap__1518535869.png")), 0).T / 255.
#print(np.abs(np.sum(np.subtract(e, s))))

def resize(image, size):
    return ImageOps.fit(image, size, Image.ANTIALIAS)


#im = Image.open('images/full snap.png')
#mat = get_matrix(im)
#mat = normalize_matrix(mat)
#mat = np.array(mat)
#print(b'#'+b16encode(bytes((238, 228, 218))))
    


