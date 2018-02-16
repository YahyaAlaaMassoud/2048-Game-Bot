from deep_neural_net_class import AgentMind
import sys, os
sys.path.append(os.path.abspath(os.path.join('.', 'bot')))
from bot import Bot
#from web_controller import WebController
from random import random, randint
import numpy as np
import smtplib
from grab import screen_grab
import time
import pickle

class GeneticEvolution():
    def __init__(self, retain = 0.3, random_select = 0.08, mutate_rate = 0.08):
        self.retain = retain
        self.random_select = random_select
        self.mutate_rate = mutate_rate
        
    def generate_init_population(self, population_size, layer_dims):
        return [AgentMind(layer_dims) for x in range(population_size)]
    
    def calculate_fitness(self, agent_score, agent_moves):
        return agent_score# / agent_moves
    
#   agents is an array of tuples of (agent, score)
    def sort_agents_by_fitness(self, agents):
        return sorted(agents, key = lambda x : x[1], reverse = True)
    
#   agents is an array of tuples of (agent, score)
    def get_fittest_agents(self, agents):
        retain_length = int(len(agents) * self.retain)
        print('retain length: ' + str(retain_length))
        fittest = agents[:retain_length]
        rest = agents[retain_length:]
        randomly_selected = self.random_selection(rest)
        fittest_agents = fittest + randomly_selected
        print('len of fittest + random selection: ' + str(len(fittest_agents)))
        return fittest_agents
    
#   agents is an array of tuples of (agent, score)
    def random_selection(self, agents):
        randomly_selected = []
        for agent in agents:
            if self.random_select > random():# and agent[1] != 0.:
                randomly_selected.append(agent)
        return randomly_selected
    
#   selected_agents is an array of AgentMinds
    def generate_new_population(self, population_size, layer_dims, selected_agents):
        agents_left = population_size - len(selected_agents)
        children = []
        while len(children) < agents_left:
            male_position = randint(0, len(selected_agents) - 1)
            female_position = randint(0, len(selected_agents) - 1)
            if male_position != female_position:
                child_params = self.crossover(selected_agents[male_position], selected_agents[female_position])
                child = AgentMind(layer_dims)
                child.set_params(child_params)
                if self.mutate_rate > random():
                    self.mutate(child)
                children.append(child)
        for agent in selected_agents:
            if self.mutate_rate > random():
                self.mutate(agent)
        new_agents = selected_agents + children
#        print('pop size: ' + str(population_size))
#        print('len of selected agents: ' + str(len(selected_agents)))
#        print('len of new agents: ' + str(len(new_agents)))
#        print()
        return new_agents
    
    def crossover(self, male, female):
        child = {}
        male_params = male.get_params()
        female_params = female.get_params()
        keys = [key for key in male_params.keys()]
        for key in keys:
            shape = male_params[key].shape
            male_dna = male_params[key].flatten()
            female_dna = female_params[key].flatten()
            cut = len(male_dna) / 2
            new_dna = np.concatenate((male_dna[:int(cut)], female_dna[int(cut):]), axis = 0)
            child[key] = new_dna.reshape(shape)
        return child
    
    def mutate(self, agent):
        params = agent.get_params()
        for (key, value) in params.items():
            shape = value.shape
            dna = value.flatten()
            mutate_position = randint(0, len(dna) - 1)
            dna[mutate_position] = np.random.randn()
            dna = dna.reshape(shape)
            params[key] = dna
        agent.set_params(params)
        return agent
    
    def Evolve(self, epochs, population_size, layer_dims):
        game_selectors = {
                            'restart_game_selector':   ".restart-button",
                            'get_score_selector':      ".score-container",
                            'is_game_over_selector':   ".game-over",
                            'scroll_to_game_selector': ".game-container"
                         }
        web_controller = WebController('https://gabrielecirulli.github.io/2048/', game_selectors)
        bot = Bot()
        
        agents = self.generate_init_population(population_size, layer_dims)
        
        
        all_scores = []
        
        for epoch in range(epochs):
            agents_scores = []
            scores = []
            for agent in agents:
                score, steps = bot.play_game(web_controller, agent, 20, 20)
                agents_scores.append((agent, self.calculate_fitness(int(score), int(steps))))
                scores.append(score)
                web_controller.restart_game()
              
            self.notify(epoch, scores)
            agents_sorted = []
            agents_sorted = self.sort_agents_by_fitness(agents_scores)
#            print(agents_sorted)
            
#            print('len of agents sorted in loop: ' + str(len(agents_sorted)))

            fittest_agents = []
            fittest_agents = self.get_fittest_agents(agents_sorted)
#            print(fittest_agents)
            fittest_agents = [agent[0] for agent in fittest_agents]
#            print('len of fittest agents in loop: ' + str(len(fittest_agents)))
            
#            os.makedirs("generation " + str(epoch + 1))    
#            self.saveAgent(fittest_agents[0].get_params(), "generation " +  str(epoch + 1), "fittest")
            agents = []
            agents = self.generate_new_population(population_size, layer_dims, fittest_agents)
#            print('len of new agents in loop: ' + str(len(agents)))
            
            all_scores.append(scores)
            
            
            scores = []
        return all_scores, agents
    
    def notify(self, epoch, scores):
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login('yahya.alaa.automatic@gmail.com', 'yahya.testing')
        
        msg = str(scores)
        server.sendmail('yahya.alaa.automatic@gmail.com', 'yahya.alaa.automatic@gmail.com', msg)
        server.quit()
        
    def saveAgent(self, agent_params, folder_name, file_name):
        with open(folder_name + "/" + file_name + ".pkl", "wb") as f:
            pickle.dump(agent_params, f, pickle.HIGHEST_PROTOCOL)
    
GA = GeneticEvolution()
gen, agents = GA.Evolve(50, 60, [16, 8, 4])

#def load_obj(name):
#    with open(name+'.pkl','rb') as f:
#        return pickle.load(f)
#






