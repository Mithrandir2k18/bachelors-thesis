import gym
import numpy as np

class env_wrapper:
    def __init__(self):
        self.last_obs = np.zeros(2)
        self.n_obs = 2
        self.n_actions = 2
        self.current_score = 0
        self.done = 0
        self.env_name = ""
        self.steps = 0

    def make(self, environment):
        self.env_name = environment
        if self.env_name == "OR" or self.env_name == "AND" or self.env_name == "NOT" or self.env_name == "XOR":
            return self
        else:
            return gym.make(self.env_name)


    def seed(self, seed):
        np.random.seed(seed)
        return


    def render(self):
        return

    def reset(self):
        if self.env_name == "NOT":
            self.last_obs = np.array([1, 0])
            np.random.shuffle(self.last_obs)
        else:
            self.last_obs = np.random.randint(0, 1, self.n_obs)

        self.done = 0
        self.steps = 0
        self.current_score = 0
        return self.last_obs

    def step(self, action):
        is_correct = 0
        self.steps += 1
        sum = np.sum(self.last_obs)
        if self.env_name == "NOT":
            if self.last_obs[action] != action:
                self.current_score += 1
                if self.steps >= 200:
                    self.done = 1

                self.last_obs = np.array([1, 0])
                np.random.shuffle(self.last_obs)

                return self.last_obs, 1, self.done, 0
            else:

                self.done = 1

                self.last_obs = np.array([1, 0])
                np.random.shuffle(self.last_obs)

                return self.last_obs, 0, self.done, 0

        elif self.env_name == "AND":
            if (sum == 2 and action == 1) or (sum != 2 and action == 0):
                is_correct = 1

            else:
                is_correct = 0

        elif self.env_name == "OR":
            if (sum > 0 and action == 1) or (sum == 0 and action == 0):
                is_correct = 1

            else:
                is_correct = 0

        elif self.env_name == "XOR":
            if (sum == 1 and action == 1) or (sum != 1 and action == 0):
                is_correct = 1

            else:
                is_correct = 0

        if is_correct == 1:
            self.current_score += 1
            if self.steps >= 200:
                self.done = 1

            self.last_obs = np.random.randint(0, 1, self.n_obs)

            return self.last_obs, 1, self.done, 0

        else:

            self.done = 1

            self.last_obs = np.random.randint(0, 1, self.n_obs)
            return self.last_obs, 0, self.done, 0








