from mesa.model import Model
from agent import Box, Goal, Bot, TaskManager

from mesa.space import SingleGrid
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector

from agent_collections import goals_collection, bots_collection

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
        'BBBBBBBBBBBFFFFFFGFFFBBBBFFFFB',
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
        'BFFFFFFFFFFFFFFFFFFFFFFFFFFF1B',
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
        self.add_box_from_map(desc)

        self.states = {}
        self.rewards = {}

        for state, cell in enumerate(self.grid.coord_iter()):
            a, pos = cell

            # Define states for the environment
            self.states[pos] = state

            # Define rewards for the environment
            if isinstance(a, Goal):
                self.rewards[state] = 1
            elif isinstance(a, Box):
                self.rewards[state] = -1
            else:
                self.rewards[state] = 0

        # Usar bucles for para agregar metas y bots
        for goal_id, x, y, name in goals_collection:
            self.add_goal(goal_id, x, y, name)

        for bot_id, x, y in bots_collection:
            self.add_robot(bot_id, x, y)

        # Crear una instancia de TaskManager
        self.task_manager = TaskManager(self)

        # Ejemplo de asignación de metas a bots
        self.task_manager.assign_goal_to_bot(101, "Salida")
        self.task_manager.assign_goal_to_bot(102, "Rack")
        self.task_manager.assign_goal_to_bot(103, "Banda")

        self.assign_rewards()

        reporters = {
            f"Bot{i+1}": lambda m, i=i: m.schedule.agents[i].total_return for i in range(len(self.schedule.agents))
        }
        # Data collector
        self.datacollector = DataCollector(
            model_reporters=reporters
        )

    def step(self):
        # Monitorea posibles colisiones antes de que los agentes tomen su paso
        self.task_manager.monitor_bots()

        #self.task_manager.assign_tasks_to_free_bots()

        # Train the agents in the environment
        if self.train and self._q_file is not None:
            for agent in self.schedule.agents:
                agent.train()
                self.train = False

        self.datacollector.collect(self)

        self.schedule.step()

        #self.running = not any([a.done for a in self.schedule.agents])
        self.running = True

    def add_box_from_map(self, desc: list):
        """
        Agrega cajas en las posiciones especificadas por el mapa ('B').

        Parameters:
        - desc: Mapa que describe la disposición inicial del entorno.
        """
        M, N = self.grid.height, self.grid.width
        for pos in self.grid.coord_iter():
            _, (x, y) = pos
            if desc[M - y - 1][x] == 'B':
                box = Box(int(f"1000{x}{y}"), self)
                self.grid.place_agent(box, (x, y))

    def add_robot(self, bot_id, x, y):
        """
        Agrega un robot en las coordenadas especificadas.

        Parameters:
        - bot_id: Identificador único del robot.
        - x: Coordenada x del robot.
        - y: Coordenada y del robot.
        """
        if self.grid.is_cell_empty((x, y)):
            bot = Bot(bot_id, self)
            self.grid.place_agent(bot, (x, y))
            self.schedule.add(bot)

    def add_goal(self, goal_id, x, y, goal_name):
        """
        Agrega una meta en las coordenadas especificadas con un nombre.

        Parameters:
        - goal_id: Identificador único de la meta.
        - x: Coordenada x de la meta.
        - y: Coordenada y de la meta.
        - goal_name: Nombre de la meta.
        """
        if self.grid.is_cell_empty((x, y)):
            goal = Goal(goal_id, self, name=goal_name)
            self.grid.place_agent(goal, (x, y))

            # Obtener el estado de la meta y añadirlo a goal_states
            state = self.states.get((x, y), None)
            if state is not None:
                self.goal_states.append(state)

    def assign_rewards(self):
        """
        Asigna recompensas específicas para cada bot basado en su meta asignada.
        """
        for agent in self.schedule.agents:
            if isinstance(agent, Bot):
                agent.rewards = {}
                for pos, state in self.states.items():  # Cambiar la iteración
                    cell_agents = self.grid.get_cell_list_contents(pos)
                    # Si es la meta asignada, recompensa positiva
                    if any(isinstance(a, Goal) and a.name == agent.target_goal_name for a in cell_agents):
                        agent.rewards[state] = 1
                    # Si es otra meta, recompensa negativa
                    elif any(isinstance(a, Goal) and a.name != agent.target_goal_name for a in cell_agents):
                        agent.rewards[state] = -2
                    # Caso general
                    elif any(isinstance(a, Box) for a in cell_agents):
                        agent.rewards[state] = -1
                    else:
                        agent.rewards[state] = 0
