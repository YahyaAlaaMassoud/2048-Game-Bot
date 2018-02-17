import win32api
import time
import grab
import numpy as np
from web_controller import WebController
from deep_neural_net_class import AgentMind
import matplotlib.pyplot as plt

class Bot():
    def __init__(self):
#        self.game_selectors = {
#                                'restart_game_selector': ".restart-button",
#                                'get_score_selector': ".score-container",
#                                'is_game_over_selector': ".game-over"
#                              }
        
        self.direction_codes= { 
                                'left' :0x25,
                                'up'   :0x26,
                                'right':0x27,
                                'down' :0x28
                              }
#        self.web_controller = WebController('https://gabrielecirulli.github.io/2048/', self.game_selectors)
       
    def play_game(self, controller, agent, shape_x, shape_y):
        time.sleep(1)
        params = agent.get_params()
        number_of_moves = 0
        number_of_invalid_moves = 0
        last_state = np.zeros((21, 1))
        consecutive_invalid = 0 
        while controller.is_game_over() == False:
            state, _ = grab.get_state(425, 230, 500, (shape_x, shape_y))
            grid_state = state
            
#            print(state.reshape(4, 4).T)
#            print()
            
            valid_up = int(self.check_valid_up(state.reshape(4, 4).T))
            valid_down = int(self.check_valid_down(state.reshape(4, 4).T))
            valid_left = int(self.check_valid_left(state.reshape(4, 4).T))
            valid_right = int(self.check_valid_right(state.reshape(4, 4).T))
            
            valid_moves = []
            valid_moves.append(valid_up)
            valid_moves.append(valid_down)
            valid_moves.append(valid_left)
            valid_moves.append(valid_right)
            
#            print(valid_moves)
            
            valid_moves = np.array(valid_moves).reshape(1, 4)
            state = np.concatenate((state, valid_moves), axis = 1)
            
            invalid_moves = np.array(0.).reshape(1, 1)
            if number_of_moves != 0:
                invalid_moves[0, 0] = number_of_invalid_moves / number_of_moves
                
#            print(invalid_moves)
#            print()
#            print('invalid ' + str(number_of_invalid_moves))
#            print('moves ' + str(number_of_moves))
#            print()
            state = np.concatenate((state, invalid_moves), axis = 1)
            
            state = state.T
            
            best_move = agent.get_best_move(state, params)
            self.perform_move(best_move)
            
            number_of_moves = number_of_moves + 1
            
            if np.array_equal(last_state, grid_state):
                number_of_invalid_moves = number_of_invalid_moves + 1
                consecutive_invalid = consecutive_invalid + 1
                last_state = grid_state
                if consecutive_invalid == 75:
                    break
                continue
            else:
                consecutive_invalid = 0
                last_state = grid_state
            time.sleep(0.4)
        return controller.get_score(), number_of_moves
    
    def perform_move(self, direction):
        win32api.keybd_event(self.direction_codes[direction], 0, 0, 0)
        
    def check_valid_up(self, mat):
        for i in range(4):
            num = False
            for j in reversed(range(4)):
                if mat[j, i] != 0.:
                    num = True
                if (j != 0 and mat[j, i] != 0. and mat[j - 1, i] == mat[j, i]) or (mat[j, i] == 0. and num == True):
                    return True
        return False
                    
    def check_valid_down(self, mat):
        for i in range(4):
            num = False
            for j in range(4):
                if mat[j, i] != 0.:
                    num = True
                if (j != 0 and mat[j, i] != 0. and mat[j - 1, i] == mat[j, i]) or (mat[j, i] == 0. and num == True):
                    return True
        return False
        
    def check_valid_left(self, mat):
        for i in range(4):
            num = False
            for j in reversed(range(4)):
                if mat[i, j] != 0.:
                    num = True
                if (j != 0. and mat[i, j] != 0. and mat[i, j - 1] == mat[i, j]) or (mat[i, j] == 0. and num == True):
                    return True
        return False
        
    def check_valid_right(self, mat):
        for i in range(4):
            num = False
            for j in range(4):
                if mat[i, j] != 0.:
                    num = True
                if (j != 0. and mat[i, j] != 0. and mat[i, j - 1] == mat[i, j]) or (mat[i, j] == 0 and num == True):
                    return True
        return False
            
#a = np.array([[ 0.,  0.,  0.,  0.],
#              [ 0. , 0. , 0.,  1.],
#              [ 0. , 0.  ,1. , 1.],
#              [ 0. , 0. , 0.  ,0.]])
#b = np.array([[ 0.,  0.,  0.,  0.],
#              [ 0. , 0. , 0.,  1.],
#              [ 0. , 0.  ,1. , 1.],
#              [ 0. , 0. , 0.  ,0.]])
#print(a != b)
##bot = Bot()
#bot.check_valid_right(a)
        
#b = np.zeros((1, 4))
#c = np.zeros((1, 3))
#d = np.concatenate((b, c), axis = 1)



#a = np.array(3).reshape(1,1)
#print(a[0,0])



