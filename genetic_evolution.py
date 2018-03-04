from agent_mind import AgentMind
from bot import Bot
from web_controller import WebController
from random import random, randint, shuffle
import numpy as np
from helper_functions import notify_by_email
import time
import pickle
import glob, os, sys
import matplotlib.pyplot as plt

class GeneticEvolution():
    def __init__(self, mutate_rate = 0.5, elitism_rate = 0.2):
        self.mutate_rate = mutate_rate
        self.elitism_rate = elitism_rate
        self.worst_fitness = 0.
        
    def generate_init_population(self, population_size, layer_dims):
        return [AgentMind(layer_dims) for x in range(population_size)]
    
    def calculate_fitness(self, agent_score, current_max_score, best_score, max_tile):
        return 100. * (float(agent_score / current_max_score) + float(agent_score / best_score) + max_tile)
    
    def sort_agents_by_fitness(self, agents = []):
        return sorted(agents, key = lambda x : x.get_fitness(), reverse = True)
    
    def tournament_selection(self, agents, k):
        tournament_agents = []

#        print('k: ' + str(k))
        
        for i in range(k):
            index = randint(0, len(agents) - 1)
            tournament_agents.append(agents[index])
            
        tournament_agents = set(tournament_agents)
        tournament_agents = self.sort_agents_by_fitness(tournament_agents)
        
        return tournament_agents[0]
    
    def generate_new_population(self, population_size, layer_dims, agents = []):
        
        self.worst_fitness = max(self.worst_fitness, agents[len(agents) - 1].get_fitness())
        coupled = {}
        children = []
        top_agents = agents[:int(len(agents) * self.elitism_rate)]
        for agent in top_agents:
            children.append(agent)
        
        while len(children) < population_size:
#            r = randint(3, 8)
            r = int(len(agents) * 0.07)
            male, female = self.tournament_selection(agents, r), self.tournament_selection(agents, r)
            if coupled.get((male, female)) == None and male != female:
#                print('crossover    -->   male: ' + str(male.get_fitness()) + ' female: ' + str(female.get_fitness()))
                new_offspring = self.crossover(male, female, layer_dims)
                coupled[(male, female)] = True
                coupled[(female, male)] = True
                if self.mutate_rate > random():
                    new_offspring = self.mutate(new_offspring)
                children.append(new_offspring)
#            elif coupled.get((male, female)) != None:
#                print('already coupled')
#            elif male == female:
#                print('same')
        
#        print('pop size: ' + str(population_size))
#        print('len of selected agents: ' + str(len(agents)))
#        print('len of new agents: ' + str(len(children)))
#        print()
        
        return children
    
    def crossover(self, male, female, layer_dims):
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
        r = randint(0, 1)
        
        if r == 0:
            for i in range(len(all_male_params)):
                if random() > 0.5:
                    new_dna.append(all_male_params[i])
                else:
                    new_dna.append(all_female_params[i])
        else:
            i, j = 0, 0
            while i >= j:
                i = randint(0, len(all_male_params) - 1)
                j = randint(0, len(all_male_params) - 1)
            new_dna = all_male_params[:i] + all_female_params[i:j] + all_male_params[j:]
            
        last_idx = 0
        
        for dim in dims:
            child[dim[0]] = np.array(new_dna[last_idx : last_idx + dim[1][0] * dim[1][1]]).reshape(dim[1])
            last_idx = last_idx + dim[1][0] * dim[1][1]
            
        child_agent = AgentMind(layer_dims)
        child_agent.set_params(child)
        
        return child_agent
    
    def mutate(self, agent):
#        print('mutate')
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
        
        all_scores = []
        best_score = 0
        
        for epoch in range(epochs):
            
            agents_scores = []
            agents_fitness = []
            scores = []
            i = 0
            current_max_score = 0
            current_max_tile = 0
            best_fitness = 0.
            
            for agent in agents:
                score, steps, max_tile, accumulated_fitness, fitness_norm = bot.play_game(web_controller, agent)
                agents_fitness.append((agent, score, max_tile))
                current_max_tile = max(current_max_tile, int(max_tile))
                best_fitness = max(best_fitness, int(100 * (accumulated_fitness / steps)))
                                
                scr = ""
                for c in score:
                    if c.isdigit():
                        scr = scr + c
                    else:
                        break
                score = int(scr)
                current_max_score = max(current_max_score, score)
                best_score = max(best_score, score)

                agent.set_fitness((float(fitness_norm / steps) ** 2) * score)
#                print('score" ' + str(score) + ' fitness: ' + str(float(fitness_norm / steps) * score))
                agents_scores.append(agent)
                scores.append((score, max_tile, agent.get_fitness()))
#                self.save_agent(agent, "fittest/[21,10,4]/2 " + str(score) + ' ' + str(i) + ' ' + str(epoch) + ".pkl")
                web_controller.restart_game()
                i = i + 1

                
            notify_by_email(epoch, scores, current_max_tile)
            
            agents_sorted = []
            agents_sorted = self.sort_agents_by_fitness(agents_scores)
            
            agents = []
            agents = self.generate_new_population(population_size, layer_dims, agents_sorted)
                        
            all_scores.append(scores)
            self.save_agent(scores, "fittest/2-3-2018/" + str(epoch) + " scores.pkl")
            self.save_agent(agents, "fittest/2-3-2018/agents " + str(epoch) + ".pkl")
            
            scores = []
        return all_scores, agents
    
    def save_agent(self, agent, path):
        with open(path, "wb") as f:
            pickle.dump(agent, f, pickle.HIGHEST_PROTOCOL)
            
    def load_agent(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)
    
    
GA = GeneticEvolution()

agents = GA.load_agent('fittest/2-3-2018/agents 2.pkl')

agents = agents[:100]

gen, agents_2 = GA.Evolve(10, 100, [21,10,4], agents, False)














