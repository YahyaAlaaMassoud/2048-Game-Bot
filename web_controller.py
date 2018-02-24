from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException
import time
import numpy as np

class WebController():
    def __init__(self, game_url, selectors = {}, open_game = True):
        self.selectors = selectors
        chrome_driver_path = "chromedriver.exe"
        options = webdriver.ChromeOptions()
        user_profile = "D:/Automation Profile"
        options.add_argument("user-data-dir=" + user_profile)
        options.add_argument("--ignore-certificate-errors")
        options.add_argument("--test-type")
        options.add_argument("--start-maximized");
        self.driver = webdriver.Chrome(chrome_driver_path, port = 0, options = options)
        if open_game == True:
            self.open_game(game_url)
            self.scroll_to_game()
    def open_game(self, url):
        self.driver.get(url)
    def scroll_to_game(self):
        element = self.driver.find_element_by_css_selector(self.selectors['scroll_to_game_selector'])
        for i in range(500):
            element.location_once_scrolled_into_view
    def restart_game(self):
        restart_button = self.driver.find_element_by_css_selector(self.selectors['restart_game_selector'])
        restart_button.click()
#        self.scroll_to_game()
    def get_score(self):
        score = self.driver.find_element_by_css_selector(self.selectors['get_score_selector'])
        return score.text
    def is_game_over(self):
        try:
            check_game_over = self.driver.find_element_by_css_selector(self.selectors['is_game_over_selector'])
            if len(check_game_over.text) != 0:
                return True
        except NoSuchElementException:
            return False
    def get_grid(self):
        grid = np.zeros(shape=(4,4), dtype='uint16')
        for x in self.driver.find_elements_by_class_name('tile'):
            cl = x.get_attribute('class').split()
            for t in cl:
                if t.startswith('tile-position-'):
                    pos = int(t[14])-1, int(t[16])-1
                elif t.startswith('tile-') and t[5].isdigit():
                    v = int(t[5:])
            grid[pos[1], pos[0]] = v
        grid = grid.reshape(1, len(grid.flatten()))
        max_value = np.max(grid)
        full_grid = grid
        grid = np.log2(grid) / np.log2(np.max(grid))
        grid[grid <= 0] = 0
        grid_list = grid.reshape(len(grid.flatten()), 1).tolist()
        new_grid = []
        for i in range(4):
            for j in range(4):
                new_grid.append(grid_list[i + j * 4])
        grid = np.array(new_grid).reshape(1, len(new_grid))
        return grid, full_grid, max_value
    def close_game(self):
        self.driver.close()
        
#        Class Test
#game_selectors = {
#                    'restart_game_selector': ".restart-button",
#                    'get_score_selector': ".score-container",
#                    'is_game_over_selector': ".game-over",
#                    'scroll_to_game_selector': ".game-container"
#                 }
#    
#controller = WebController('https://gabrielecirulli.github.io/2048/', game_selectors)










