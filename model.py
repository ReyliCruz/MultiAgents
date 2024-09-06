from mesa.model import Model
from agent import Box, Goal, Bot, TaskManager

from mesa.space import SingleGrid, MultiGrid
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector

from agent_collections import goals_collection, bots_collection, articles_collection, chargers_collection

import numpy as np
import random
from queue import Queue
import os
import json


class Environment(Model):
    DEFAULT_MODEL_DESC = [
        'BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB',
        'BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB',
        'BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB',
        'BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB',
        'BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB',
        'BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB',
        'BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB',
        'BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB',
        'BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB',
        'BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB',
        'BBBBBBBBBBBFFFFFFFFFBBBBBBBBBB',
        'BBBBBBBBBBBFFFFFFFFFBBBBBBBBBB',
        'BBBBBBBBBBBFFFFFFFFFBBBBBBBBBB',
        'BBBBBBBBBBBFFFFFFFFFBBBBBBBBBB',
        'BBBBBBBBBBBFFFFFFFFFBBBBFFFFBB',
        'BBBBBBBBBBBFFFFFBBFFBBBBFFFFBB',
        'BBBBBBBBBBBBFFFFBBFFBBBBFFFFBB',
        'BBBBBBBBBBBBBFFFFFFFFFFFFFFFBB',
        'BBBBBBBBBBBBBFFFFFFFFFFFFFFFBB', #'BBBBBBBBBBBBBFFFBBFBBFFFFFFFBB',
        'BBBBBBBBBBBFFFFFBBFFFFFFFFFFBB',
        'BFFFFFFFFFFFFFFFFFFFFFFFFFFFBB',
        'BFFFFFFFFFFFFFFFFFFFFFFFFFFFFB',
        'BFFFFFFFFFFFFFFBBBBBFFFFFFFFFB',
        'BFFFFFFFFFFFFFFBBBBBFFFFFFFFFB',
        'BFFFFFFFFFFFFFFFFFFFFFFFFFFFFB',
        'BFFFFFFFFFFFFFFFFFFFFFFFFFFFFB',
        'BFFFFFFFFFFFFFFFFFFFFFFFFFFFFB',
        'BFFFFFFFFFFFFFFFFFFFFFFFFFFFFB',
        'BFFFFFFFFFFBBBBBBBBBBBBBFFFFFB',
        'BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB',
    ]

    def __init__(self, desc=None, q_file=None, train=False):
        super().__init__()
        self._q_file = q_file
        self.data = {"robots": []}  # Inicialización de la estructura de datos
        self.steps = 0

        self.goal_states = []
        self.articles_queue = Queue()  # Cola para almacenar artículos
        self.time_counter = 0  # Contador de tiempo para extraer artículos
        self.next_generation_time = 1  # Tiempo inicial aleatorio para extraer artículo
        self.bot_teams = []
        self.total_deliverables = 0
        self.total_stored = 0
        self.selected_articles = []
        self.total_energy_cost = 0

        self.free_chargers = Queue()
        for charger in chargers_collection:
            self.free_chargers.put(charger)

        # Default environment description for the model
        self.train = train
        if desc is None:
            desc = self.DEFAULT_MODEL_DESC

        M, N = len(desc), len(desc[0])

        #self.grid = SingleGrid(M, N, False)
        self.grid = MultiGrid(M, N, False)
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

        for goal_id, x, y, name in goals_collection:
            self.add_goal(goal_id, x, y, name)

        for bot_id, x, y in bots_collection:
            self.add_robot(bot_id, x, y)

        self.task_manager = TaskManager(self)

        reporters = {
            f"Bot{i+1}": lambda m, i=i: m.schedule.agents[i].total_return for i in range(len(self.schedule.agents))
        }
        # Data collector
        self.datacollector = DataCollector(
            model_reporters=reporters
        )

    def collect_robot_data(self):
        """Registra la posición actual y la batería de los robots en cada paso."""
        if self.steps == 0:
            # Inicializa el JSON solo en el primer paso
            for agent in self.schedule.agents:
                if isinstance(agent, Bot):
                    robot_info = {
                        "spawnPosition": {
                            "x": agent.pos[0],
                            "y": agent.pos[1]
                        },
                        "path": []
                    }
                    self.data["robots"].append(robot_info)

        for i, agent in enumerate(self.schedule.agents):
            if isinstance(agent, Bot):
                # Registrar el camino y la batería en cada paso
                robot_step_info = {
                    "x": agent.pos[0],
                    "y": agent.pos[1],
                    "battery": agent.battery
                }
                self.data["robots"][i]["path"].append(robot_step_info)

    def update_json(self):
        """Guarda el archivo JSON con todos los registros cuando la simulación termina."""
        with open("robot_data.json", "w") as json_file:
            json.dump(self.data, json_file, indent=4)

    def save_summary_to_json(self):
        """Guarda un resumen de los datos en un archivo JSON separado."""
        summary_data = {
            "total_deliverables": self.total_deliverables,
            "total_stored": self.total_stored,
            "total_energy_cost": self.total_energy_cost,
            "steps": self.steps
        }

        with open("simulation_summary.json", "w") as summary_file:
            json.dump(summary_data, summary_file, indent=4)

    def step(self):
        self.time_counter += 1

        if self.time_counter >= self.next_generation_time:
            self.time_counter = 0
            self.next_generation_time = random.randint(10, 15)
            self.generate_and_queue_article_extended_version()

        self.task_manager.assign_tasks_to_free_bots_extended_version()

        self.assign_goals_to_bots_extended_version()

        self.task_manager.manage_bot_movements()

        self.datacollector.collect(self)

        self.schedule.step()
        
        
        self.collect_robot_data()  # Colectar los datos de los robots

        self.steps += 1
        
        total_deliverables = 0
        total_stored = 0
        total_energy_cost = 0
        for agent in self.schedule.agents:
            if isinstance(agent, Bot):  
                total_deliverables += agent.robot_total_deliverable
                total_stored += agent.robot_total_stored
                total_energy_cost += agent.robot_total_battery_cost


        self.running = True

        
        # Verificar si se ha alcanzado el paso máximo (100)
        if total_deliverables >= 10:
            print("Pedidos entregados, fin de la simulación.")
            self.total_deliverables = total_deliverables
            self.total_stored = total_stored
            self.total_energy_cost = total_energy_cost
            self.steps = self.steps

            self.update_json()    # Guardar los datos en el archivo JSON
            self.save_summary_to_json()
            self.running = False  # Detener la simulación

        # Asegurar que la simulación continúe corriendo si no ha terminado
        else:
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
        #if self.grid.is_cell_empty((x, y)):
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

            #self.states[goal_name] = (x, y)

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

    def generate_and_queue_article(self):
        """Genera un artículo aleatorio de la colección y lo agrega a la cola."""
        if articles_collection:
            # Seleccionar un artículo aleatorio de la colección
            article = random.choice(articles_collection)
            self.selected_articles.append(article)
            self.articles_queue.put(article)  # Agregar el artículo a la cola
            
    def generate_and_queue_article_extended_version(self):
        """Genera un artículo aleatorio de la colección y lo agrega a la cola si tanto el origen como el destino no están asignados."""
        if articles_collection:
            while True:  # Bucle hasta encontrar un artículo con origen y destino nuevos
                # Seleccionar un artículo aleatorio de la colección
                article = random.choice(articles_collection)
                article_id, weight, origin_name, destination_name = article

                # Obtener las coordenadas del origen y destino
                origin_coords = next(((x, y) for (goal_id, x, y, name) in goals_collection if name == origin_name), None)
                destination_coords = next(((x, y) for (goal_id, x, y, name) in goals_collection if name == destination_name), None)


                # Verificar si el origen o destino ya están asignados a algún bot
                origin_in_use = False
                destination_in_use = False
                
                for agent in self.schedule.agents:
                    if isinstance(agent, Bot):
                        # Verificar si el bot tiene asignado el mismo origen o destino
                        if agent.target_goal_name == origin_name or agent.target_goal_name == destination_name:
                            origin_in_use = True
                            destination_in_use = True
                            break  # Si ya está asignado, no es necesario verificar más

                        # Verificar si algún bot está físicamente en el origen o destino
                        if agent.pos == origin_coords:
                            origin_in_use = True
                            break
                        if agent.pos == destination_coords:
                            destination_in_use = True
                            break

                # Si tanto el origen como el destino están libres, agregar el artículo a la cola
                if not origin_in_use and not destination_in_use:
                    self.selected_articles.append(article)
                    self.articles_queue.put(article)  # Agregar el artículo a la cola
                    print(f"Artículo {article_id} generado y asignado: origen {origin_name}, destino {destination_name}")
                    break  # Salir del bucle cuando se encuentra un artículo válido
                else:
                    print(f"Artículo {article_id} descartado: origen {origin_name}, destino {destination_name} ya están en uso.")
                    break


    def assign_goals_to_bots(self):
        """
        Asigna metas a los bots en función de su tarea actual.
        Si el bot no tiene una meta asignada, se le asigna el origen.
        Si el bot ya ha llegado al origen, se le asigna el destino.
        """
        for agent in self.schedule.agents:
            if isinstance(agent, Bot) and agent.task:
                article_id, weight, origin_name, destination_name = agent.task

                # Obtener las coordenadas del origen y destino a partir de los nombres
                origin_coords = next(((x, y) for (goal_id, x, y, name) in goals_collection if name == origin_name), None)
                destination_coords = next(((x, y) for (goal_id, x, y, name) in goals_collection if name == destination_name), None)

                '''
                if (agent.charger_name == ""):
                    agent.charger_name = self.free_chargers.get()
                elif (agent.battery >= 100):
                    self.free_chargers.put(agent.charger_name)
                    agent.charger_name = ""
                '''
                    

                if(agent.aux_target != ""):
                    self.task_manager.assign_goal_to_bot(agent.unique_id, agent.charger_name)

                    self.assign_rewards()

                    # Verificar si el archivo de Q-values existe
                    if os.path.exists(f"./q_values{agent.target_goal_name}.npy"):
                        agent.q_file = f"q_values{agent.target_goal_name}.npy"
                        agent.training_step = agent.MAX_NUM_TRAINING_STEPS  # Evitar entrenamiento adicional
                        agent.load_q_values(agent.q_file)
                    else:
                        print(f"Archivo ./q_values{agent.target_goal_name}.npy no encontrado. Entrenando agente.")
                        agent.training_step = 0
                        agent.train()  # Entrenar al agente si no existe el archivo de Q-values

                else:
                    if agent.target_goal_name == "" and agent.done == False:
                        if agent.pos == origin_coords:
                            # Si el bot ya está en el origen, asignar destino
                            self.task_manager.assign_goal_to_bot(agent.unique_id, destination_name)
                        else:
                            # Si no, asignar origen
                            self.task_manager.assign_goal_to_bot(agent.unique_id, origin_name)

                        self.assign_rewards()

                        # Verificar si el archivo de Q-values existe
                        if os.path.exists(f"./q_values{agent.target_goal_name}.npy"):
                            agent.q_file = f"q_values{agent.target_goal_name}.npy"
                            agent.training_step = agent.MAX_NUM_TRAINING_STEPS  # Evitar entrenamiento adicional
                            agent.load_q_values(agent.q_file)
                        else:
                            print(f"Archivo ./q_values{agent.target_goal_name}.npy no encontrado. Entrenando agente.")
                            agent.training_step = 0
                            agent.train()  # Entrenar al agente si no existe el archivo de Q-values

                        #agent.train()


    def assign_goals_to_bots_extended_version(self):
        """
        Asigna metas a los bots en función de su tarea actual.
        Si el bot no tiene una meta asignada, se le asigna el origen.
        Si el bot ya ha llegado al origen, se le asigna el destino.
        """
        for agent in self.schedule.agents:
            if isinstance(agent, Bot) and agent.task:
                article_id, weight, origin_name, destination_name = agent.task

                # Obtener las coordenadas del origen y destino a partir de los nombres
                origin_coords = next(((x, y) for (goal_id, x, y, name) in goals_collection if name == origin_name), None)
                destination_coords = next(((x, y) for (goal_id, x, y, name) in goals_collection if name == destination_name), None)

                
                if (agent.charger_name == ""):
                    agent.charger_name = self.free_chargers.get()
                elif (agent.battery >= 100):
                    self.free_chargers.put(agent.charger_name)
                    agent.charger_name = ""
                

                if(agent.aux_target != ""):
                    self.task_manager.assign_goal_to_bot(agent.unique_id, agent.charger_name)

                    self.assign_rewards()

                    # Verificar si el archivo de Q-values existe
                    if os.path.exists(f"./q_values{agent.target_goal_name}.npy"):
                        agent.q_file = f"q_values{agent.target_goal_name}.npy"
                        agent.training_step = agent.MAX_NUM_TRAINING_STEPS  # Evitar entrenamiento adicional
                        agent.load_q_values(agent.q_file)
                    else:
                        print(f"Archivo ./q_values{agent.target_goal_name}.npy no encontrado. Entrenando agente.")
                        agent.training_step = 0
                        agent.train()  # Entrenar al agente si no existe el archivo de Q-values

                else:
                    # Verificar si el bot está en equipo y si está en las coordenadas del origen
                    if agent.in_team_mode and agent.pos == origin_coords:
                        # Buscar el equipo del bot
                        team_members = next((team for team in self.bot_teams if agent in team), [])
                        
                        # Si todos los miembros del equipo están en el origen, asignarles el destino
                        if all([bot.pos == next(((x, y) for (goal_id, x, y, name) in goals_collection if name == bot.task[2]), None) for bot in team_members]):
                            for bot in team_members:
                                bot.team_formation = True
                                self.task_manager.assign_goal_to_bot(bot.unique_id, destination_name)
                            print(f"Equipo con el líder {agent.unique_id} ha llegado al origen. Asignando destino.")

                            self.assign_rewards()

                            # Verificar si el archivo de Q-values existe
                            if os.path.exists(f"./q_values{agent.target_goal_name}.npy"):
                                agent.q_file = f"./q_values{agent.target_goal_name}.npy"
                                agent.training_step = agent.MAX_NUM_TRAINING_STEPS
                                agent.load_q_values(agent.q_file)
                            else:
                                print(f"Archivo ./q_values{agent.target_goal_name}.npy no encontrado. Entrenando agente.")
                                agent.training_step = 0
                                agent.train()

                        else:
                            # Si no, esperar en el origen
                            print(f"Bot {agent.unique_id} está esperando en el origen para su equipo.")
                            continue  # No hacer nada hasta que todo el equipo esté en el origen
                    else:
                        # Si no está en equipo o no está en el origen, proceder normalmente
                        if agent.target_goal_name == "" and not agent.done:
                            if agent.pos == origin_coords:
                                # Si el bot ya está en el origen, asignar destino
                                self.task_manager.assign_goal_to_bot(agent.unique_id, destination_name)
                            else:
                                # Si no, asignar origen
                                self.task_manager.assign_goal_to_bot(agent.unique_id, origin_name)

                            self.assign_rewards()

                            # Verificar si el archivo de Q-values existe
                            if os.path.exists(f"./q_values{agent.target_goal_name}.npy"):
                                agent.q_file = f"./q_values{agent.target_goal_name}.npy"
                                agent.training_step = agent.MAX_NUM_TRAINING_STEPS
                                agent.load_q_values(agent.q_file)
                            else:
                                print(f"Archivo ./q_values{agent.target_goal_name}.npy no encontrado. Entrenando agente.")
                                agent.training_step = 0
                                agent.train()
                        
                        elif agent.target_goal_name != "" and agent.battery >= 95:
                            self.task_manager.assign_goal_to_bot(agent.unique_id, agent.target_goal_name)
                            
                            self.assign_rewards()

                            # Verificar si el archivo de Q-values existe
                            if os.path.exists(f"./q_values{agent.target_goal_name}.npy"):
                                agent.q_file = f"./q_values{agent.target_goal_name}.npy"
                                agent.training_step = agent.MAX_NUM_TRAINING_STEPS
                                agent.load_q_values(agent.q_file)
                            else:
                                print(f"Archivo ./q_values{agent.target_goal_name}.npy no encontrado. Entrenando agente.")
                                agent.training_step = 0
                                agent.train()