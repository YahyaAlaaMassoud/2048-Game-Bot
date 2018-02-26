import numpy as np
from helper_functions import relu, softmax, get_best_move_direction, sigmoid

class AgentMind():
    def __init__(self, layers_dims):
        self.__layers_dims = layers_dims
        self.__parameters = self.init_params_he(layers_dims)
        self.__fitness = 0
        
    def get_params(self):
        return self.__parameters
    
    def set_params(self, new_params):
        self.__parameters = new_params
        
    def get_fitness(self):
        return self.__fitness
    
    def set_fitness(self, fitness):
        self.__fitness = fitness

    def init_params_he(self, layers_dims):
        parameters = {}
        L = len(layers_dims)
        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2. / layers_dims[l - 1]) #/ np.sqrt(layers_dims[l - 1])#* np.sqrt(2. / layers_dims[l - 1])
#            parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
            parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))#np.random.randn(layers_dims[l], 1)
        return parameters
    
    def linear_forward(self, A, W, b):
        Z = W.dot(A) + b
        return Z
    
    def activate_forward(self, A_prev, W, b, activation):
        if activation == "softmax":
            Z = self.linear_forward(A_prev, W, b)
            A, _ = softmax(Z)
        elif activation == "relu":
            Z = self.linear_forward(A_prev, W, b)
            A, _ = relu(Z)
        return A
    
    def get_best_move(self, X, parameters):
        A = X
        L = len(parameters) // 2    
        for l in range(1, L):
            A_prev = A 
            A = self.activate_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
        AL = self.activate_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "softmax")
        AL[AL == np.max(AL)] = 1
        AL[AL != np.max(AL)] = 0
        return get_best_move_direction(AL)
    
