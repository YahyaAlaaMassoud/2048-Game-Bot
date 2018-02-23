from deep_neural_net_class import AgentMind
from bot import Bot
from web_controller import WebController
from random import random, randint, shuffle
import numpy as np
import smtplib
from grab import screen_grab
import time
import pickle

class GeneticEvolution():
    def __init__(self, retain = 0.2, random_select = 0.05, mutate_rate = 0.03):
        self.retain = retain
        self.random_select = random_select
        self.mutate_rate = mutate_rate
        
    def generate_init_population(self, population_size, layer_dims):
        return [AgentMind(layer_dims) for x in range(population_size)]
    
    def calculate_fitness(self, agent_score):
        return agent_score
    
#   agents is an array of tuples of (agent, score)
    def sort_agents_by_fitness(self, agents):
        return sorted(agents, key = lambda x : x[1], reverse = True)
    
#   agents is an array of tuples of (agent, score)
    def get_fittest_agents(self, agents):
        retain_length = int(len(agents) * self.retain)
        fittest = []
        fittest = agents[:retain_length]
        rest = []
        rest = agents[retain_length:]
        randomly_selected = []
        randomly_selected = self.random_selection(rest)
        fittest_agents = []
        fittest_agents = fittest + randomly_selected
        return fittest_agents
    
#   agents is an array of tuples of (agent, score)
    def random_selection(self, agents):
        randomly_selected = []
        for agent in agents:
            if self.random_select > random():
                randomly_selected.append(agent)
        return randomly_selected
    
#   selected_agents is an array of AgentMinds
    def generate_new_population(self, population_size, layer_dims, selected_agents):
        agents_left = population_size - len(selected_agents)
        children = []
        children.append(selected_agents[0])
        top_agents = len(selected_agents) // 3
        for i in range(top_agents):
            for j in range(i, top_agents):
                child_params = self.crossover(selected_agents[i], selected_agents[j])
                child = AgentMind(layer_dims)
                child.set_params(child_params)
                children.append(child)
        while len(children) < agents_left:
            male_position = randint(0, len(selected_agents) - 1)
            female_position = randint(0, len(selected_agents) - 1)
            if male_position != female_position:
                child_params = self.crossover(selected_agents[male_position], selected_agents[female_position])
                child = AgentMind(layer_dims)
                child.set_params(child_params)
                children.append(child)
        new_agents = []
        new_agents = selected_agents + children
        for agent in new_agents:
            if self.mutate_rate > random():
                agent = self.mutate(agent)
        
        print('pop size: ' + str(population_size))
        print('len of selected agents: ' + str(len(selected_agents)))
        print('len of new agents: ' + str(len(new_agents)))
        print()
        return new_agents
    
#    uniform crossover
    def crossover(self, male, female):
        child = {}
        male_params = male.get_params()
        female_params = female.get_params()
        dims = []
        all_male_params = []
        all_female_params = []
        keys = [key for key in male_params.keys()]
        for key in keys:
            dims.append((key, male_params[key].shape))
            male_dna = male_params[key].flatten()
            female_dna = female_params[key].flatten()
            all_male_params = all_male_params + male_dna.tolist()
            all_female_params = all_female_params + female_dna.tolist()
        new_dna = []
        for i in range(len(all_male_params)):
            if random() > 0.5:
                new_dna.append(all_male_params[i])
            else:
                new_dna.append(all_female_params[i])
        last_idx = 0
        for dim in dims:
            child[dim[0]] = np.array(new_dna[last_idx : last_idx + dim[1][0] * dim[1][1]]).reshape(dim[1])
            last_idx = last_idx + dim[1][0] * dim[1][1]
        return child
    
    def mutate(self, agent):
        params = agent.get_params()
        dims = []
        all_agent_params = []
        keys = [key for key in params.keys()]
        for key in keys:
            dims.append((key, params[key].shape))
            agent_dna = params[key].flatten()
            all_agent_params = all_agent_params + agent_dna.tolist()
        r = randint(0, 3)
        if r == 0:
            mutate_position = randint(0, len(all_agent_params) - 1)
            all_agent_params[mutate_position] = np.random.randn()
        elif r == 1:
            i, j = 0, 0
            while i == j:
                i = randint(0, len(all_agent_params) - 1)
                j = randint(0, len(all_agent_params) - 1)
            all_agent_params[i], all_agent_params[j] = all_agent_params[j], all_agent_params[i]
        elif r == 2:
            i, j = 0, 0
            while i >= j:
                i = randint(0, len(all_agent_params) - 1)
                j = randint(0, len(all_agent_params) - 1)
            cpy = all_agent_params[i:j]
            shuffle(cpy)
            all_agent_params[i:j] = cpy
        else:
            i, j = 0, 0
            while i >= j:
                i = randint(0, len(all_agent_params) - 1)
                j = randint(0, len(all_agent_params) - 1)
            cpy = all_agent_params[i:j][::-1]
            all_agent_params[i:j] = cpy
        last_idx = 0
        for dim in dims:
            param = np.array(all_agent_params[last_idx : last_idx + dim[1][0] * dim[1][1]]).reshape(dim[1])
            last_idx = last_idx + dim[1][0] * dim[1][1]
            params[dim[0]] = param
        agent.set_params(params)
        return agent
    
    def Evolve(self, epochs, population_size, layer_dims, old_agents = [], generate_population = True):
        game_selectors = {
                            'restart_game_selector':   ".restart-button",
                            'get_score_selector':      ".score-container",
                            'is_game_over_selector':   ".game-over",
                            'scroll_to_game_selector': ".game-container"
                         }
        web_controller = WebController('https://gabrielecirulli.github.io/2048/', game_selectors)
        time.sleep(60)
        bot = Bot()
        
        if generate_population == True:
            agents = self.generate_init_population(population_size, layer_dims)
        else:
            agents = old_agents
            
        good_agent_1 = self.load_agent("fittest", "796")[0]
        good_agent_2 = self.load_agent("fittest", "1444")[0]
        good_agent_3 = self.load_agent("fittest", "3880")[0]
        
        agents.append(good_agent_1)
        agents.append(good_agent_2)
        agents.append(good_agent_3)
        

        all_scores = []
        
        for epoch in range(epochs):
            
            agents_scores = []
            scores = []
            maximum_value = 0
            i = 0
            
            for agent in agents:
                score, steps, max_value = bot.play_game(web_controller, agent, 20, 20)
                maximum_value = max(maximum_value, int(max_value))
                scr = ""
                for c in score:
                    if c.isdigit():
                        scr = scr + c
                    else:
                        break

                self.save_agent((agent, scr), "fittest", str(scr) + ' ' + str(i) + ' ' + str(epoch))
                agents_scores.append((agent, self.calculate_fitness(int(scr))))
                scores.append(scr)
                web_controller.restart_game()
                i = i + 1
              
            self.notify(epoch, scores, maximum_value)
            agents_sorted = []
            agents_sorted = self.sort_agents_by_fitness(agents_scores)
            
            
            fittest_agents = []
            fittest_agents = self.get_fittest_agents(agents_sorted)
            fittest_agents = [agent[0] for agent in fittest_agents]
            
            
            agents = []
            agents = self.generate_new_population(population_size, layer_dims, fittest_agents)
                        
            all_scores.append(scores)
            
            
            scores = []
        return all_scores, agents
    
    def notify(self, epoch, scores, maximum_value):    
        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login('yahya.alaa.automatic@gmail.com', 'yahya.testing')
            
            maxi = self.get_maximum_fitness(scores)
            avg = self.get_average_fitness(scores)
            
            msg = str(scores) + ' \nlen: ' + str(len(scores)) + '\n max value: ' + str(maximum_value) + ' \n max: ' + str(maxi) + ' \n avg: ' + str(avg)
            server.sendmail('yahya.alaa.automatic@gmail.com', 'yahya.alaa.automatic@gmail.com', msg)
            server.quit()
        except Exception:
            print('error happened while sending email ' + str(maxi) + ' ' + str(avg))
            
        
    def get_average_fitness(self, scores):
        summation = 0
        for s in scores:
            summation = summation + int(s)
        return float(summation / len(scores))
    
    def get_maximum_fitness(self, scores):
        maxi = 0
        for s in scores:
            maxi = max(maxi, int(s))
        return maxi
        
    def save_agent(self, agent_params, folder_name, file_name):
        with open(folder_name + "/" + file_name + ".pkl", "wb") as f:
            pickle.dump(agent_params, f, pickle.HIGHEST_PROTOCOL)
            
    def load_agent(self, folder_name, file_name):
        with open(folder_name + "/" + file_name + ".pkl", "rb") as f:
            return pickle.load(f)
    
    
GA = GeneticEvolution()

gen, agents = GA.Evolve(50, 110, [21, 8, 4])



