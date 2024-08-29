from mesa.model import Model
from agent import Box, Goal, Bot

from mesa.space import SingleGrid
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector

import numpy as np


class Environment(Model):
    DEFAULT_MODEL_DESC = [
        'BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB',
        'BBBBBBBBBBBFFBBFBBFBBBBBBFFFFB',
        'BBBBBBBBBBBFFBBFBBFBBBBBBFFFFB',
        'BBBBBBBBBBBFFFFFFFFFFBBBBFFFFB',
        'BBBBBBBBBBBFFFFFFFFFFBBBBFFFFB',
        'BBBBBBBBBBBFFFFFFFFFFBBBBFFFFB',
        'BBBBBBBBBBBFFFFFFFFFFBBBBFFFFB',
        'BBBBBBBBBBBFFFFFFFFFFBBBBFFFFB',
        'BBBBBBBBBBBFFFFFFBBFFBBBBFFFFB',
        'BBBBBBBBBBBBBBFFFBBFFFFFFFFFFB',
        'BBBBBBBBBBBBBBFFFFFFFFFFFFFFFB',
        'BBBBBBBBBBBBBBFFFBBFBBFFBBFFFB',
        'BFFFFFFFFFFFFFFFFBBFBBFFBBFFFB',
        'BFFFFFFFFFFFFFFFFFFFFFFFFFFFFB',
        'BFFFFFFFFFFFFFFFFFFFFFFFFFFFFB',
        'BFFFFFFFFFFFFFFFBBBBBBFBBFFFFB',
        'BFFFFFFFFFFFFFFFBBBBBBFBBFFFFB',
        'BFFFFFFFFFFFFFFFFFFFFFFFFFFFFB',
        'BFFFFFFFFFFFFFFFFFFFFFFFFFFFFB',
        'BFFFFFFFFFFFFFFFFFFFFFFFFFFFFB',
        'BFFFFFFFFFFFFFFFFFFFFFFFFFFFFB',
        'BFFFFFFFFFFFFFFFFFFFFFFFFFFFFB',
        'BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB',
        'BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB',
        'BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB',
        'BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB',
        'BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB',
        'BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB',
        'BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB',
        'BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB'
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
                if a.name == "Salida":  # Aquí verificamos si es el objetivo que busca
                    self.rewards[state] = 1  # Recompensa positiva si es el objetivo correcto
                else:
                    self.rewards[state] = -1  # Recompensa negativa si no es el objetivo correcto
                self.goal_states.append(state)
            elif isinstance(a, Box):
                self.rewards[state] = -1
            else:
                self.rewards[state] = 0

        # Llamar a la función para agregar metas adicionales con nombres
        additional_goals = [
            (3, 10, "Salida"),
            (10, 10, "Rack"),
            (15, 15, "Banda")
        ]  
        
        # Ejemplo de coordenadas y nombres para metas adicionales
        self.add_goals(additional_goals)
        bot_details = [
            (0, (28, 8), "Salida"),  # Bot con ID 1, en posición (1, 1), con objetivo "Salida"
            (1, (27, 8), "Rack"),    # Bot con ID 2, en posición (2, 2), con objetivo "Rack"
            (2, (26, 8), "")    # Bot con ID 3, en posición (3, 3), con objetivo "Banda"
        ]

        self.add_bots(bot_details)
        
        reporters = {
            f"Bot{i+1}": lambda m, i=i: m.schedule.agents[i].total_return for i in range(len(self.schedule.agents))
        }
        # Data collector
        self.datacollector = DataCollector(
            model_reporters=reporters
        )

    def add_goals(self, goal_details):
        """
        Add additional goals to the environment at specified coordinates with specified names.
        
        Parameters:
        - goal_details: List of tuples, each containing (x, y, name) representing the coordinates and name of the new goals.
        """
        for coord in goal_details:
            x, y, name = coord
            if self.grid.is_cell_empty((x, y)):
                goal = Goal(int(f"10{x}{y}"), self, name=name)
                self.grid.place_agent(goal, (x, y))

                # Update states and rewards
                state = self.states[(x, y)]
                if name == "Salida":
                    self.rewards[state] = 1  # Assign positive reward for the correct goal
                else:
                    self.rewards[state] = -1  # Assign negative reward for incorrect goals
                self.goal_states.append(state)
            else:
                print(f"La celda {(x, y)} no está vacía. No se puede colocar una meta aquí.")

    def add_bots(self, bot_details):
        """
        Inicializa y coloca bots en el entorno en posiciones específicas con objetivos específicos.

        Parameters:
        - bot_details: Lista de tuplas, cada una conteniendo (id, (x, y), target_goal_name) representando el
                       identificador único del bot, la posición inicial y el nombre del objetivo.
        """
        for bot_id, (x, y), goal_name in bot_details:
            if self.grid.is_cell_empty((x, y)):
                bot = Bot(bot_id, self, target_goal_name=goal_name)
                self.grid.place_agent(bot, (x, y))
                self.schedule.add(bot)
                print(f"Bot {bot_id} colocado en posición ({x}, {y}) con objetivo: {goal_name}")
            else:
                print(f"La celda ({x}, {y}) no está vacía. No se puede colocar el bot aquí.")

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
                goal_name = f"Goal_{x}_{y}"
                meta = Goal(int(f"10{x}{y}"), self, name=goal_name)
                self.grid.place_agent(meta, (x, y))
            else:
                try:
                    bot_num = int(desc[M - y - 1][x])
                    target_goal_name = "Salida"
                    bot = Bot(int(f"{bot_num}"), self, self._q_file, target_goal_name=target_goal_name)
                    self.grid.place_agent(bot, (x, y))
                    self.schedule.add(bot)

                except ValueError:
                    pass
