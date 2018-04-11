import numpy as np
import random

class ExperienceReplay():
    def __init__(self, max_size = 50):
        self.experiences = []
        self.max_size = max_size
        
    def add_experience(self, experience):
        self.experiences.append(experience)
        if len(self.experiences) > self.max_size:
            self.experiences = self.experiences[len(self.experiences) - self.max_size:]

    def sample(self, size):
        return random.sample(self.experiences, size)