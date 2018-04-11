from keras.models import Sequential
from keras.layers import Dense
from keras.losses import mean_squared_error
import numpy as np
from experience_replay import ExperienceReplay
from bot import Bot
from web_controller import WebController
import time
import grab
from helper_functions import get_action_name, get_action_number

class DQL():
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate = 0.9, epsilon_min = 0.01, epsilon = 1.0, epsilon_decay = 0.995, episodes = 100, rand_steps = 100, max_experience = 10000, update_freq = 10):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.episodes = episodes
        self.rand_steps = rand_steps
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_experience = max_experience
        self.epsilon_min = epsilon_min 
        self.learning_rate = learning_rate
        self.update_freq = update_freq
        self.model = self.create_model()
        self.experience = ExperienceReplay(self.max_experience)
    
    def create_model(self):
        #model.predict(np.random.rand(1, 16))
        model = Sequential()
        model.add(Dense(units = self.hidden_dim, kernel_initializer = 'uniform', activation = 'relu', input_dim = self.input_dim))
        model.add(Dense(units = self.output_dim, kernel_initializer = 'uniform', activation = 'linear'))
        model.compile(optimizer = 'adam', loss = mean_squared_error, metrics = ['accuracy'])
        return model
    
    def play_game(self):
        game_selectors = {
                            'restart_game_selector':   ".restart-button",
                            'get_score_selector':      ".score-container",
                            'is_game_over_selector':   ".game-over",
                            'scroll_to_game_selector': ".game-container"
                         }
        web_controller = WebController('https://gabrielecirulli.github.io/2048/', game_selectors)
        time.sleep(40)
        bot = Bot()
        
        total_steps = 0
        rewards = []
        
        for episode in range(self.episodes):
            state, _, _ = grab.get_state(425, 230, 500, (20,20))
            total_reward = 0.
            while web_controller.is_game_over() == False:
                total_steps += 1
                score = self.to_int(web_controller.get_score())
                if np.random.rand() < self.epsilon:
                    action = np.random.randint(0, self.output_dim)
                else:
                    action = np.argmax(self.model.predict(state))
                action = get_action_name(action)
                bot.perform_move(action)
                time.sleep(0.1)
                new_state, full_state, max_value = grab.get_state(425, 230, 500, (20,20))
                new_score = self.to_int(web_controller.get_score())
                reward = new_score - score#np.sum(full_state / 2048.)# * 10.
                terminate = web_controller.is_game_over()
                if terminate == True: 
                    reward = -10
                if str(state) == str(new_state):
                    reward = -1
                self.experience.add_experience((state, new_state, action, reward, terminate))
                
                if total_steps > self.rand_steps:
                    if total_steps % self.update_freq == 0:
                        batch_size = 32
                        if len(self.experience.experiences) < 32:
                            batch_size = len(self.experience.experiences)
                        experiences = self.experience.sample(batch_size)
                        for s, new_s, a, r, t in experiences:
                            target_reward = r
                            if not t:
                                target_reward = r + self.learning_rate * np.amax(self.model.predict(new_s))
                            desired_reward = self.model.predict(s)
                            desired_reward[0][get_action_number(a)] = target_reward
                            self.model.fit(s, desired_reward, epochs = 1, verbose = 0)

                state = new_state
                total_reward += reward
                
            print('reward: ' + str(total_reward))
            rewards.append(total_reward)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            web_controller.restart_game()
            time.sleep(0.3)
        return rewards
    
    def to_int(self, score):
        scr = ""
        for c in score:
            if c.isdigit():
                scr = scr + c
            else:
                break
        return int(scr)


dql = DQL(16, 10, 4)
r = dql.play_game()

