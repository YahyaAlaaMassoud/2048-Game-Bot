from deep_neural_net_class import AgentMind
from bot import Bot
from web_controller import WebController
from random import random, randint
import numpy as np

class GeneticEvolution():
    def __init__(self, retain = 0.3, random_select = 0.08, mutate_rate = 0.08):
        self.retain = retain
        self.random_select = random_select
        self.mutate_rate = mutate_rate
        
    def generate_init_population(self, population_size, layer_dims):
        return [AgentMind(layer_dims) for x in range(population_size)]
    
    def calculate_fitness(self, agent_score, agent_moves):
        return agent_score / agent_moves
    
#   agents is an array of tuples of (agent, score)
    def sort_agents_by_fitness(self, agents):
        return sorted(agents, key = lambda x : x[1], reverse = True)
    
#   agents is an array of tuples of (agent, score)
    def get_fittest_agents(self, agents):
        retain_length = int(len(agents) * self.retain)
        fittest = agents[:retain_length]
        rest = agents[retain_length:]
        randomly_selected = self.random_selection(rest)
        return fittest + randomly_selected
    
#   agents is an array of tuples of (agent, score)
    def random_selection(self, agents):
        randomly_selected = []
        for agent in agents:
            if self.random_select > random() and agent[1] != 0.:
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
        return selected_agents + children
    
    def crossover(self, male, female):
        child = {}
        male_params = male.get_params()
        female_params = female.get_params()
        for (male_key, male_value), (female_key, female_value) in zip(male_params.items(), female_params.items()):
            shape = male_value.shape
            male_dna = male_value.flatten()
            female_dna = female_value.flatten()
            cut = randint(0, len(male_dna) - 1)
            new_dna = np.append(male_dna[:int(cut)], female_dna[int(cut):])
            child[male_key] = new_dna.reshape(shape)
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
        agents_scores = []
        scores = []
        all_scores = []
        
        for epoch in range(epochs):
            for agent in agents:
                score, steps = bot.play_game(web_controller, agent)
                agents_scores.append((agent, self.calculate_fitness(int(score), int(steps))))
                scores.append(score)
                print(score)
                web_controller.restart_game()
            
            agents_sorted = self.sort_agents_by_fitness(agents_scores)
            
            fittest_agents = self.get_fittest_agents(agents_sorted)
            fittest_agents = [agent[0] for agent in fittest_agents]
            
            agents = self.generate_new_population(population_size, layer_dims, fittest_agents)
            all_scores.append(scores)
            scores = []
        return all_scores
    
GA = GeneticEvolution()
gen = GA.Evolve(4, 10, [25 * 25 * 3, 930, 4])



