import numpy as np
import smtplib
import pickle

def get_best_move_direction(decision):
    if decision[0, 0] == 1:
        return 'up'
    if decision[1, 0] == 1:
        return 'down'
    if decision[2, 0] == 1:
        return 'left'
    if decision[3, 0] == 1:
        return 'right'
    
def get_action_name(decision):
    if decision == 0:
        return 'up'
    if decision == 1:
        return 'down'
    if decision == 2:
        return 'left'
    if decision == 3:
        return 'right'
    
def get_action_number(decision):
    if decision == 'up':
        return 0
    if decision == 'down':
        return 1
    if decision == 'left':
        return 2
    if decision == 'right':
        return 3
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x)), (x)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis = 0), (x)

def relu(x):
    return np.maximum(0, x), (x)

def notify_by_email(epoch, scores, maximum_value):
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login('YOU EMAIL', 'YOUR PASSWORD')
        msg = str(epoch) + '\n' + str(scores) + ' \nlen: ' + str(len(scores)) + '\n max tile: ' + str(maximum_value)
        server.sendmail('yahya.alaa.automatic@gmail.com', 'yahya.alaa.automatic@gmail.com', msg)
        server.quit()
    except Exception as ex:
        print('Error happened while sending email.\n' + str(ex))
        
def get_average_fitness(scores):
    return np.average(scores)

def get_maximum_fitness(scores):
    return np.max(scores)

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

def save_file(agent, path):
    with open(path, "wb") as f:
        pickle.dump(agent, f, pickle.HIGHEST_PROTOCOL)
        
def load_file(path):
    with open(path, "rb") as f:
        return pickle.load(f)