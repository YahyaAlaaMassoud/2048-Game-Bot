from agent_mind import AgentMind
from bot import Bot
from web_controller import WebController
from random import random, randint, shuffle
import numpy as np
from helper_functions import notify_by_email, save_file, load_file
import time
import glob, os, sys, getopt
import matplotlib.pyplot as plt

class GeneticEvolution():
    def __init__(self, mutate_rate = 0.05, elitism_rate = 0.4, crossover_operator = "random", mutation_operator = "random"):
        self.mutate_rate = mutate_rate
        self.elitism_rate = elitism_rate
        self.crossover_operator = crossover_operator
        self.mutation_operator = mutation_operator
        
    def generate_init_population(self, population_size, layer_dims):
        return [AgentMind(layer_dims) for x in range(population_size)]
    
    def calculate_fitness(self, strategy_fitness, steps, score):
        return (float(strategy_fitness / steps) ** 2) * score
    
    def sort_agents_by_fitness(self, agents = []):
        return sorted(agents, key = lambda x : x.get_fitness(), reverse = True)
    
    def tournament_selection(self, agents, k):
        tournament_agents = []

        for i in range(k):
            index = randint(0, len(agents) - 1)
            tournament_agents.append(agents[index])
            
        tournament_agents = set(tournament_agents)
        tournament_agents = self.sort_agents_by_fitness(tournament_agents)
        
        return tournament_agents[0]
    
    def generate_new_population(self, population_size, layer_dims, agents = []):
        
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
        if self.crossover_operator == "random":
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
                
        elif self.crossover_operator == "uniform":
            for i in range(len(all_male_params)):
                if random() > 0.5:
                    new_dna.append(all_male_params[i])
                else:
                    new_dna.append(all_female_params[i])
                    
        elif self.crossover_operator == "two-point":
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
            
        if self.mutation_operator == "random":
            r = randint(0, 3)
            if r == 0:
                all_agent_params = self.mutate_resetting(all_agent_params)
            elif r == 1:
                all_agent_params = self.mutate_swap(all_agent_params)
            elif r == 2:
                all_agent_params = self.mutate_scramble(all_agent_params)
            else:
                all_agent_params = self.mutate_inverse(all_agent_params)
        
        elif self.mutation_operator == "resetting":
            all_agent_params = self.mutate_resetting(all_agent_params)
            
        elif self.mutation_operator == "swap":
            all_agent_params = self.mutate_swap(all_agent_params)
        
        elif self.mutation_operator == "scramble":
            all_agent_params = self.mutate_scramble(all_agent_params)
            
        elif self.mutation_operator == "inverse":
            all_agent_params = self.mutate_inverse(all_agent_params)
            
        last_idx = 0
        
        for dim in dims:
            param = np.array(all_agent_params[last_idx : last_idx + dim[1][0] * dim[1][1]]).reshape(dim[1])
            last_idx = last_idx + dim[1][0] * dim[1][1]
            params[dim[0]] = param
            
        agent.set_params(params)
        
        return agent
    
    def mutate_resetting(self, params):
        mutate_position = randint(0, len(params) - 1)
        params[mutate_position] = np.random.randn()
        return params
    
    def mutate_swap(self, params):
        i, j = 0, 0
        while i == j:
            i = randint(0, len(params) - 1)
            j = randint(0, len(params) - 1)
        params[i], params[j] = params[j], params[i]
        return params
    
    def mutate_scramble(self, params):
        i, j = 0, 0
        while i >= j:
            i = randint(0, len(params) - 1)
            j = randint(0, len(params) - 1)
        cpy = params[i:j]
        shuffle(cpy)
        params[i:j] = cpy
        return params
    
    def mutate_inverse(self, params):
        i, j = 0, 0
        while i >= j:
            i = randint(0, len(params) - 1)
            j = randint(0, len(params) - 1)
        cpy = params[i:j][::-1]
        params[i:j] = cpy
        return params
        
    
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
                score, steps, max_tile, fitness = bot.play_game(web_controller, agent)
                agents_fitness.append((agent, score, max_tile))
                
                current_max_tile = max(current_max_tile, int(max_tile))
                best_fitness = max(best_fitness, int(100 * (fitness / steps)))

                score = self.to_int(score)
                
                current_max_score = max(current_max_score, score)
                best_score = max(best_score, score)

                agent.set_fitness(self.calculate_fitness(fitness, steps, score))
                
                agents_scores.append(agent)
                scores.append((score, max_tile, agent.get_fitness()))
                
                web_controller.restart_game()
                i += 1
                
#            notify_by_email("YOUR EMAIL", "PASSWORD", epoch, scores, current_max_tile)
            
            agents_sorted = []
            agents_sorted = self.sort_agents_by_fitness(agents_scores)
            
            agents = []
            agents = self.generate_new_population(population_size, layer_dims, agents_sorted)
                        
            all_scores.append(scores)
            save_file(scores, "fittest/11-3-2018/" + str(epoch) + " scores.pkl")
            save_file(agents, "fittest/11-3-2018/agents " + str(epoch) + ".pkl")
            
            scores = []
        return all_scores, agents
    
    def to_int(self, score):
        scr = ""
        for c in score:
            if c.isdigit():
                scr = scr + c
            else:
                break
        return int(scr)
    
def main(argv):
    mutate_rate = 0.05
    elitism_rate = 0.4
    crossover_operator = "random"
    mutation_operator = "random"
    epochs = 100
    population_size = 50
    layer_dims = [21, 10, 4]
    generate_population = True
    old_agents = []
    
    try:
        opts, args = getopt.getopt(argv,"h",["mr=","er=","co=","mo=","epochs=","psz=","hl="])
    except getopt.GetoptError:
        print('python genetic_evolution.py --mr <mutate_rate> --er <elitism_rate> --co <crossover_operator> --mo <mutation_operator> --epochs <epochs> --psz <population_size> --hl <hidden_layer_units>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('python genetic_evolution.py --mr <mutate_rate> --er <elitism_rate> --co <crossover_operator> --mo <mutation_operator> --epochs <epochs> --psz <population_size> --hl <hidden_layer_units>')
            sys.exit()
        elif opt in ("--mr"):
            mutate_rate = float(arg)
        elif opt in ("--er"):
            elitism_rate = float(arg)
        elif opt in ("--co"):
            crossover_operator = (arg)
        elif opt in ("--mo"):
            mutation_operator = (arg)
        elif opt in ("--epochs"):
            epochs = int(arg)
        elif opt in ("--psz"):
            population_size = int(arg)
        elif opt in ("--hl"):
            layer_dims = [21, int(arg), 4]
    print(mutate_rate)
    print(elitism_rate)
    print(crossover_operator)
    print(mutation_operator)
    print(epochs)
    print(population_size)
    print(layer_dims)
    
    GA = GeneticEvolution(mutate_rate, elitism_rate, crossover_operator, mutation_operator)
    _,_ = GA.Evolve(epochs, population_size, layer_dims)


if __name__ == "__main__":
    main(sys.argv[1:])
