from mesa.model import Model
from agent import Box, Goal, Bot

from mesa.space import SingleGrid
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector

import numpy as np


class Environment(Model):
    DEFAULT_MODEL_DESC = [
        'BBBBBBBBBBBBBBBBBBBB',
        'B13FFFFFFFFFFFFFF42B',
        'BFFFFFFFFFFFFFFFFFFB',
        'BFFFFFFFFFFFFFFFFFFB',
        'BFFFFFFFFFFFFFFFFFFB',
        'BFFFFFFFFFFFFFFFFFFB',
        'BFFFFFFFFFFFFFFFFFFB',
        'BFFFFFFFFFFFFFFFFFFB',
        'BFFFFFFFFFFFFFFFFFFB',
        'BFFFFFFFFFFFFFFFFFFB',
        'BFFFFFFFFGFFFFFFFFFB',
        'BFFFFFFFFFFFFFFFFFFB',
        'BFFFFFFFFFFFFFFFFFFB',
        'BFFFFFFFFFFFFFFFFFFB',
        'BFFFFFFFFFFFFFFFFFFB',
        'BFFFFFFFFFFFFFFFFFFB',
        'BFFFFFFFFFFFFFFFFFFB',
        'BFFFFFFFFFFFFFFFFFFB',
        'BFFFFFFFFFFFFFFFFFFB',
        'BBBBBBBBBBBBBBBBBBBB'
    ]

    def __init__(self, desc=None, q_file=None, train=False):
        super().__init__()
        self._q_file = q_file

        self.goal_states = []

        # Default environment description for the model
        self.train = train
        if desc is None:
            desc = self.DEFAULT_MODEL_DESC

        M, N = len(desc), len(desc[0])

        self.grid = SingleGrid(M, N, False)
        self.schedule = SimultaneousActivation(self)

        # Place agents in the environment
        self.place_agents(desc)

        self.states = {}
        self.rewards = {}
        for state, cell in enumerate(self.grid.coord_iter()):
            a, pos = cell

            # Define states for the environment
            self.states[pos] = state

            # Define rewards for the environment
            if isinstance(a, Goal):
                self.rewards[state] = 1
                self.goal_states.append(state)
            elif isinstance(a, Box):
                self.rewards[state] = -1
            else:
                self.rewards[state] = 0

        reporters = {
            f"Bot{i+1}": lambda m, i=i: m.schedule.agents[i].total_return for i in range(len(self.schedule.agents))
        }
        # Data collector
        self.datacollector = DataCollector(
            model_reporters=reporters
        )

    def step(self):
        # Train the agents in the environment
        if self.train and self._q_file is not None:
            for agent in self.schedule.agents:
                agent.train()
                self.train = False

        self.datacollector.collect(self)

        self.schedule.step()

        self.running = not any([a.done for a in self.schedule.agents])

    def place_agents(self, desc: list):
        M, N = self.grid.height, self.grid.width
        for pos in self.grid.coord_iter():
            _, (x, y) = pos
            if desc[M - y - 1][x] == 'B':
                box = Box(int(f"1000{x}{y}"), self)
                self.grid.place_agent(box, (x, y))
            elif desc[M - y - 1][x] == 'G':
                meta = Goal(int(f"10{x}{y}"), self)
                self.grid.place_agent(meta, (x, y))
            else:
                try:
                    bot_num = int(desc[M - y - 1][x])
                    bot = Bot(int(f"{bot_num}"), self, self._q_file)
                    self.grid.place_agent(bot, (x, y))
                    self.schedule.add(bot)

                except ValueError:
                    pass
