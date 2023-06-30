import gymnasium as gym
import copy
import pygame
from pygame import gfxdraw
from gymnasium.error import DependencyNotInstalled

class Environment:

    def __init__(self, env, rendering=None, noise=None, converter=None, **kwargs):
        """
        A custom environment that allows for easy change in the
        physical properties in the environment and addition of 
        noise to the observations
        """
        self.env = gym.make(env, **kwargs)
        self.original_params = self.env.unwrapped.__dict__
        self.noise = noise
        self.converter = converter

        if rendering == 'dual' or rendering == 'single' or rendering == 'overlay':
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
        elif rendering == 'overlay':
            self.rendering = 'overlay'
            self.screen = pygame.display.set_mode((self.env.unwrapped.screen_dim, self.env.unwrapped.screen_dim))
            self.screen_left = pygame.Surface((self.env.unwrapped.screen_dim, self.env.unwrapped.screen_dim))
            self.clock_left = pygame.time.Clock()
            self.screen_right = pygame.Surface((self.env.unwrapped.screen_dim, self.env.unwrapped.screen_dim))
            self.screen_left.set_alpha(128)
            self.screen_right.set_alpha(32)
            self.clock_right = pygame.time.Clock()
            pygame.display.set_caption('Overlay display mode')
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
        #print('>>> ', self.noise)
        if self.noise is not None:
            noisy_state = self.noise(self.env.state)
            noisy_observation = self.converter(noisy_state)
            self.render(noisy_state)
        else:
            self.render()
        return observation, reward, terminated, truncated, info

    def render(self, noisy_state=None):
        if self.rendering == 'dual' or self.rendering == 'overlay':
            self.render_one_screen(self.screen_left, self.clock_left, 0, 0)

            if noisy_state is not None:
                old_state = self.env.unwrapped.state
                self.env.unwrapped.state = noisy_state
                if self.rendering == 'dual':
                    self.render_one_screen(self.screen_right, self.clock_right, self.env.unwrapped.screen_dim, 0)
                else:
                    self.render_one_screen(self.screen_right, self.clock_right, 0, 0)
                self.env.unwrapped.state = old_state
            else:
                if self.rendering == 'dual':
                    self.render_one_screen(self.screen_right, self.clock_right, self.env.unwrapped.screen_dim, 0)
                else:
                   self.render_one_screen(self.screen_right, self.clock_right, 0, 0) 
            
            self._pygame_flip()
        
        elif self.rendering == 'single':
            self._render()
            self._pygame_flip()

    def render_one_screen(self, screen, clock, x, y):
        self.env.unwrapped.screen = screen
        self.env.unwrapped.clock = clock
        self._render()
        self.screen.blit(screen, (x, y))

    def reset(self, **kwargs):
        self.step_ = 0
        
        observation, info = self.env.reset(**kwargs)
        self.render()
        return observation, info
    
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()


        

