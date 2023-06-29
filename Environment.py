class Environment:

    def __init__(self, env):
        """
        A custom environment that allows for easy change in the
        physical properties in the environment and addition of 
        noise to the observations
        """
        self.env = env
        self.changes = {}
        self.step_ = 0

    def add_change(self, time_point, scheme):
        self.changes[time_point] = scheme

    def step(self, action):
        self.step_ += 1
        print('STEP: ', self.step_, self.env.unwrapped.__dict__['g'], self.env._elapsed_steps, self.env._max_episode_steps)

        if self.step_ in self.changes.keys():
            self.env.unwrapped.__dict__.update(self.changes[self.step_])

        return self.env.step(action)

    def reset(self, **kwargs):
        self.step_ = 0

        return self.env.reset(**kwargs)
    
    def close(self):

        self.env.close()

        

