import numpy as np

def get_best_move_direction(x):
    if x[0, 0] == 1:
        return 'up'
    if x[1, 0] == 1:
        return 'down'
    if x[2, 0] == 1:
        return 'left'
    if x[3, 0] == 1:
        return 'right'
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x)), (x)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis = 0), (x)

def relu(x):
    return np.maximum(0, x), (x)

#   image shape (length, height, depth)
#   output vec shape (length * height * depth, 1) 
#   len = shape[0], height = shape[1], depth = shape[2] 
#   axis = 1 -> return column vector
#   acis = 0 -> return row vector 
def img2vec(img, row_or_col = 2):
    if row_or_col == 0:
        vec = img.reshape(1, (img.shape[0] * img.shape[1] * img.shape[2]))
    elif row_or_col == 1:
        vec = img.reshape((img.shape[0] * img.shape[1] * img.shape[2]), 1)
    elif row_or_col == 2:
        vec = img.reshape((img.shape[0] * img.shape[1] * img.shape[2]))
    return vec

#def test_img2vec():
#    image = np.array([[[ 0.67826139,  0.29380381],
#            [ 0.90714982,  0.52835647],
#            [ 0.4215251 ,  0.45017551]],
#    
#           [[ 0.92814219,  0.96677647],
#            [ 0.85304703,  0.52351845],
#            [ 0.19981397,  0.27417313]],
#    
#           [[ 0.60659855,  0.00533165],
#            [ 0.10820313,  0.49978937],
#            [ 0.34144279,  0.94630077]]])
#    print ((img2vec(image, 2).shape))
#test_img2vec()