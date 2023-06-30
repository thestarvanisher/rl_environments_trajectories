import gymnasium as gym
import copy
import pygame
from pygame import gfxdraw
from gymnasium.error import DependencyNotInstalled

class Environment:

    def __init__(self, env, rendering=None, noise=None, **kwargs):
        """
        A custom environment that allows for easy change in the
        physical properties in the environment and addition of 
        noise to the observations
        """
        self.env = gym.make(env, **kwargs)
        self.original_params = self.env.unwrapped.__dict__
        self.noise = noise

        if rendering == 'dual' or rendering == 'single':
            try:
                import pygame
                from pygame import gfxdraw
            except ImportError:
                raise DependencyNotInstalled("pygame is not installed, run `pip install gym[classic_control]`")

            self._pygame_flip = pygame.display.flip
            pygame.display.flip = self.do_nothing

            self._render = self.env.unwrapped.render
            self.env.unwrapped.render = self.do_nothing

        if rendering == 'dual':
            self.rendering = 'dual'
            self.screen = pygame.display.set_mode((self.env.unwrapped.screen_dim * 2, self.env.unwrapped.screen_dim))
            self.screen_left = pygame.Surface((self.env.unwrapped.screen_dim, self.env.unwrapped.screen_dim))
            self.clock_left = pygame.time.Clock()
            self.screen_right = pygame.Surface((self.env.unwrapped.screen_dim, self.env.unwrapped.screen_dim))
            self.clock_right = pygame.time.Clock()
            pygame.display.set_caption('Dual display mode')
        elif rendering == 'single':
            self.rendering = 'single'
            self.screen = pygame.display.set_mode((self.env.unwrapped.screen_dim, self.env.unwrapped.screen_dim))
            self.env.unwrapped.screen = self.screen
            pygame.display.set_caption('Single display mode')

        self.changes = {}
        self.step_ = 0

    def do_nothing(self):
        pass

    def add_change(self, time_point, scheme):
        self.changes[time_point] = scheme

    def step(self, action):
        self.step_ += 1
        print('STEP: ', self.step_, self.env.unwrapped.__dict__['g'], self.env._elapsed_steps, self.env._max_episode_steps)

        if self.step_ in self.changes.keys():
            self.env.unwrapped.__dict__.update(self.changes[self.step_])

        
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.render()
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.rendering == 'dual':
            self.env.unwrapped.screen = self.screen_left
            self.env.unwrapped.clock = self.clock_left
            self._render()
            self.screen.blit(self.screen_left, (0, 0))

            self.env.unwrapped.screen = self.screen_right
            self.env.unwrapped.clock = self.clock_right
            self._render()
            self.screen.blit(self.screen_left, (self.env.unwrapped.screen_dim, 0))

            self._pygame_flip()
        
        elif self.rendering == 'single':
            self._render()
            self._pygame_flip()


    def reset(self, **kwargs):
        self.step_ = 0
        
        observation, info = self.env.reset(**kwargs)
        self.render()
        return observation, info
    
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()


        

