import win32api
import time
import grab
import numpy as np
from web_controller import WebController
from agent_mind import AgentMind
#import gridplotlib.pyplot as plt

class Bot():
    def __init__(self):
        self.__direction_codes= { 
                                    'left' : 0x25,
                                    'up'   : 0x26,
                                    'right': 0x27,
                                    'down' : 0x28
                                }
       
    def play_game(self, controller, agent):
        time.sleep(0.3)
        params = agent.get_params()
        
        number_of_moves = 0
        number_of_invalid_moves = 0
        last_state = None
        consecutive_invalid_moves = 0 
        maximum_value = 0
        fitness = 0.
        
        while controller.is_game_over() == False:
#            state, full_state, max_value = grab.get_state(545, 370, 500, (20,2))
            state, full_state, max_value = grab.get_state(425, 230, 500, (20,20))
            maximum_value = max(maximum_value, int(max_value))
            grid_state = state
            
            valid_up = int(self.check_valid_up(state.reshape(4, 4).T))
            valid_down = int(self.check_valid_down(state.reshape(4, 4).T))
            valid_left = int(self.check_valid_left(state.reshape(4, 4).T))
            valid_right = int(self.check_valid_right(state.reshape(4, 4).T))
            
            valid_moves = []
            valid_moves.append(valid_up)
            valid_moves.append(valid_down)
            valid_moves.append(valid_left)
            valid_moves.append(valid_right)
            
            valid_moves = np.array(valid_moves).reshape(1, 4)
            state = np.concatenate((state, valid_moves), axis = 1)
            
            invalid_moves = np.array(0.).reshape(1, 1)
            if number_of_moves != 0:
                invalid_moves[0, 0] = number_of_invalid_moves / number_of_moves
                
            state = np.concatenate((state, invalid_moves), axis = 1)
            
            state = state.T
            
            best_move = agent.get_best_move(state, params)
            self.perform_move(best_move)
            
            number_of_moves = number_of_moves + 1
            
            if np.array_equal(last_state, grid_state):
                number_of_invalid_moves = number_of_invalid_moves + 1
                consecutive_invalid_moves = consecutive_invalid_moves + 1
                last_state = grid_state
                if consecutive_invalid_moves == 50:
                    break
                continue
            else:
                consecutive_invalid_moves = 0
                last_state = grid_state

            fitness_W = np.array([[0.135759 , 0.121925  , 0.102812  , 0.099937  ],
                                  [0.0997992, 0.0888405 , 0.076711  , 0.0724143 ],
                                  [0.060654 , 0.0562579 , 0.037116  , 0.0161889 ],
                                  [0.0125498, 0.00992495, 0.00575871, 0.00335193]])
            
            grid_state = grid_state.reshape(4, 4).T
            
            fitness = fitness + np.multiply(grid_state, fitness_W).sum()
                
            time.sleep(0.135)
        return controller.get_score(), number_of_moves - number_of_invalid_moves, maximum_value, fitness
    
    def perform_move(self, direction):
        win32api.keybd_event(self.__direction_codes[direction], 0, 0, 0)
        
    def check_valid_up(self, grid):
        for i in range(4):
            num = False
            for j in reversed(range(4)):
                if grid[j, i] != 0.:
                    num = True
                if (j != 0 and grid[j, i] != 0. and grid[j - 1, i] == grid[j, i]) or (grid[j, i] == 0. and num == True):
                    return True
        return False
                    
    def check_valid_down(self, grid):
        for i in range(4):
            num = False
            for j in range(4):
                if grid[j, i] != 0.:
                    num = True
                if (j != 0 and grid[j, i] != 0. and grid[j - 1, i] == grid[j, i]) or (grid[j, i] == 0. and num == True):
                    return True
        return False
        
    def check_valid_left(self, grid):
        for i in range(4):
            num = False
            for j in reversed(range(4)):
                if grid[i, j] != 0.:
                    num = True
                if (j != 0. and grid[i, j] != 0. and grid[i, j - 1] == grid[i, j]) or (grid[i, j] == 0. and num == True):
                    return True
        return False
        
    def check_valid_right(self, grid):
        for i in range(4):
            num = False
            for j in range(4):
                if grid[i, j] != 0.:
                    num = True
                if (j != 0. and grid[i, j] != 0. and grid[i, j - 1] == grid[i, j]) or (grid[i, j] == 0 and num == True):
                    return True
        return False