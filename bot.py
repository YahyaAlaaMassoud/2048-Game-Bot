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
       
    def play_game(self, controller, agent):
        params = agent.get_params()
        number_of_moves = 1
        same = 0
        last_state = np.zeros((25 * 25 * 3, 1))
        while controller.is_game_over() == False:
            state = grab.screen_grab(425, 230, 500, (25, 25)).T
            state = state / 255.
            best_move = agent.get_best_move(state, params)
            self.perform_move(best_move)
            if np.array_equal(last_state, state):
                same = same + 1
            last_state = state
            number_of_moves = number_of_moves + 1
            if same == 20:
                break
        time.sleep(0.5)
        return controller.get_score(), number_of_moves
    
    def perform_move(self, direction):
        win32api.keybd_event(self.direction_codes[direction], 0, 0, 0)
            