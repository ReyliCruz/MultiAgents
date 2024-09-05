from mesa.agent import Agent
import numpy as np
from queue import Queue
import random
from agent_collections import goals_collection

class Bot(Agent):

    MAX_NUM_TRAINING_STEPS = 1000
    NUM_OF_ACTIONS = 4

    # Define the movements (0: down, 1: right, 2: up, 3: left)
    MOVEMENTS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.q_values = None
        self.done = False
        self.state = None
        self.next_state = None
        self.action = None
        self.next_pos = None
        self.total_return = 0
        self.training_step = 0
        self.movements = 0
        self.target_goal_name = ""
        self.previous_target = ""
        self.aux_target = ""
        self.rewards = {}
        self.isFree = True
        self.task = None
        self.q_file = None
        self.history = []
        self.battery = 999999999 #100
        self.weight_box = 0
        self.low_battery = 35
        self.charger_name = ""
        self.charging = False
        self.in_team_mode = False  # Indica si el bot está en modo equipo
        self.team_size = 0  # Tamaño del equipo (2 o 4 bots)
        self.path = []

        self.epsilon = 1.0 #0.1
        self.alpha = 0.1
        self.gamma = 0.9

        self.num_states = self.model.grid.width * self.model.grid.height

        if self.q_file is None:
            self.reset_q_values()
        else:
            self.load_q_values(self.q_file)

    def reset_q_values(self):
        self.q_values = {(state, action): np.random.uniform(0, .01)
                         for state in range(self.num_states)
                         for action in range(self.NUM_OF_ACTIONS)}

    def step(self) -> None:
        if self.battery <= 0:
            print(f"Bot {self.unique_id} sin batería, no puede moverse.")
            return
        
        if (self.task and self.battery<self.low_battery and self.aux_target == ""):
            self.deliver_or_charge()
        elif (self.target_goal_name == "" and (not self.task)) and self.battery < 100:
            self.aux_target = "_"
            

        if self.state is None:
            self.state = self.model.states[self.pos]
        

        # Agent chooses an action from the policy
        #self.action = self.eps_greedy_policy(self.state)

        #self.action = self.greedy_policy(self.state)

        '''
        self.history.append(self.pos)
        if len(self.history) > 10:
            self.history.pop(0)

        if self.detect_oscillation_all_cases():
            #self.backtrack(1)
            self.action = self.random_policy()
        else:
            self.action = self.greedy_policy(self.state)

        # Get the next position
        self.next_pos = self.perform(self.pos, self.action)
        self.next_state = self.model.states[self.next_pos]
        '''

        # Si el bot está en equipo, buscar al líder en la lista de equipos
        if self.in_team_mode:
            leader_bot = self.get_leader_from_team()

            if leader_bot and leader_bot.unique_id != self.unique_id:
                # Copiar la acción del líder y ajustar posición en el equipo
                self.action = leader_bot.action
                self.adjust_team_position(leader_bot)
            else:
                # Si el bot es el líder, usa A* para encontrar el camino al objetivo
                if self.target_goal_name:
                    goal_coords = next(((x, y) for (goal_id, x, y, name) in goals_collection if name == self.target_goal_name), None)
                    if goal_coords:
                        if not self.path or self.path[-1] != goal_coords:
                            self.path = self.a_star_team_with_retries(self.pos, goal_coords)
                        
                        if self.path:
                            self.next_pos = self.path.pop(0)  # Tomar el siguiente paso en el camino
                            self.next_state = self.model.states[self.next_pos]
                            self.action = None  # No es necesario definir acción, ya que estamos siguiendo el camino
                        else:
                            print(f"El líder {self.unique_id} no puede encontrar un camino hacia el objetivo.")


                '''
                # Si el bot es el líder, toma decisiones
                self.history.append(self.pos)
                if len(self.history) > 10:
                    self.history.pop(0)

                if self.detect_oscillation_all_cases():
                    self.action = self.random_policy()
                else:
                    self.action = self.greedy_policy(self.state)

                # Obtener la siguiente posición
                self.next_pos = self.perform(self.pos, self.action)
                self.next_state = self.model.states[self.next_pos]
                '''
        
        else:
            # Si no está en equipo, funciona normalmente
            self.history.append(self.pos)
            if len(self.history) > 10:
                self.history.pop(0)
            '''
            if self.detect_oscillation_all_cases():
                self.action = self.random_policy()
            else:
                self.action = self.greedy_policy(self.state)

            '''
            self.action = self.greedy_policy(self.state)

            # Obtener la siguiente posición
            self.next_pos = self.perform(self.pos, self.action)
            self.next_state = self.model.states[self.next_pos]
        
        
        if (self.charging):
            self.battery += 20
            if self.battery >= 100:
                self.battery = 100
                self.charging = False
                if self.aux_target != "_":
                    self.target_goal_name = self.aux_target
                self.aux_target = ""
                #self.charger_name = ""
        elif (self.target_goal_name != ""):
            self.battery = self.battery -  (1 + self.weight_box * 0.1)/2
        

    def get_leader_from_team(self):
        """
        Busca y retorna el líder de su equipo desde la lista bot_teams en el modelo.
        """
        for team in self.model.bot_teams:
            if self in team:
                return team[0]  # El primer bot en el equipo es el líder
        return None  # No pertenece a ningún equipo
    
    def get_team_from_leader(self, leader_bot):
        """
        Obtiene el equipo del líder bot desde la lista de equipos en el modelo.
        """
        for team in self.model.bot_teams:
            if leader_bot in team:
                return team
        return None

    def adjust_team_position(self, leader_bot):
        """
        Ajusta la posición del bot en relación con el líder para formar un cuadro o pareja.
        El equipo tiene el siguiente patrón:
        - Bot 0 (líder): posición original
        - Bot 1: a la derecha del líder (si el equipo tiene 2 o 4 miembros)
        - Bot 2: abajo del líder (solo si el equipo tiene 4 miembros)
        - Bot 3: en la esquina inferior derecha (solo si el equipo tiene 4 miembros)
        """
        team = self.get_team_from_leader(leader_bot)
        
        if team is not None:
            leader_pos = leader_bot.pos

            # Encontrar la posición del bot en el equipo
            index_in_team = team.index(self)

            if self.team_size == 2:
                if index_in_team == 1:  # Bot a la derecha del líder (equipo de 2 bots)
                    self.next_pos = (leader_pos[0] + 1, leader_pos[1])  # Derecha

            elif self.team_size == 4:
                if index_in_team == 1:  # Bot a la derecha del líder
                    self.next_pos = (leader_pos[0] + 1, leader_pos[1])  # Derecha
                elif index_in_team == 2:  # Bot abajo del líder
                    self.next_pos = (leader_pos[0], leader_pos[1] - 1)  # Abajo
                elif index_in_team == 3:  # Bot en la esquina inferior derecha
                    self.next_pos = (leader_pos[0] + 1, leader_pos[1] - 1)  # Derecha y abajo

            # Actualizar el siguiente estado
            self.next_state = self.model.states[self.next_pos]


    def advance(self) -> None:
        if (self.target_goal_name == "") or self.charging:
            return
        
        article_id, weight, origin, destination = self.task

        # Si el bot está en equipo, seguir al líder
        if self.in_team_mode:
            leader_bot = self.get_leader_from_team()

            if leader_bot and leader_bot.unique_id != self.unique_id:
                # Ajustar la posición relativa al líder
                self.adjust_team_position(leader_bot)
                self.model.grid.move_agent(self, self.next_pos)
                self.state = self.next_state
                return
        
        # Check if the agent can move to the next position
        if self.model.grid.is_cell_empty(self.next_pos) or (
            self.next_state in self.model.goal_states and 
            any(isinstance(goal, Goal) and goal.name == self.target_goal_name 
                for goal in self.model.grid.get_cell_list_contents(self.next_pos))
        ):
            if self.next_state in self.model.goal_states:

                if self.target_goal_name == self.charger_name and self.aux_target != "":
                    self.target_goal_name = ""
                    self.charging = True

                # Verificar si el bot ha llegado al `origin`
                if self.target_goal_name == origin and self.previous_target == "":
                    self.weight_box = weight
                    self.target_goal_name = ""
                    self.previous_target = origin

                # Verificar si el bot ha llegado al `destination`
                elif self.target_goal_name == destination and self.previous_target == origin:
                    self.weight_box = 0
                    self.done = True
                    self.target_goal_name = ""
                    self.previous_target = ""
                    self.task = None
                    self.isFree = True

            # Move the agent to the next position and update everything
            self.model.grid.move_agent(self, self.next_pos)
            self.movements += 1

            # Update the state
            self.state = self.next_state

            # Get the reward
            reward = self.model.rewards[self.next_state]

        else:
            # If the agent cannot move to the next position, the reward is -2
            reward = -2

        # Update the q-values
        self._update_q_values(self.state, self.action, reward, self.next_state)

        # Update the total return
        self.total_return += reward

    def save_q_values(self):
        np.save(f"./q_values{self.target_goal_name}.npy", self.q_values)

    def train(self):
        self.epsilon = 1.0
        #inital_pos = self.pos
        #initial_state = self.model.states[inital_pos]

        for episode in range(10000):
            training_step = 0
            done = False
            total_return = 0
            movements = 0
            #pos = inital_pos
            #state = initial_state

            # Encontrar una posición inicial aleatoria vacía
            pos = self.find_random_empty_position()
            state = self.model.states[pos]

            # Colocar al agente en la nueva posición inicial
            self.model.grid.move_agent(self, pos)
            self.state = state

            while not done:
                training_step += 1
                action = self.eps_greedy_policy(state)
                #action = self.random_policy()

                next_pos = self.perform(pos, action)
                next_state = self.model.states[next_pos]

                #reward = self.model.rewards[next_state]
                #reward = self.rewards.get(next_state, -1)
                reward = self.rewards.get(next_state, -1)

                #if next_state in self.model.goal_states:
                if next_state in self.model.goal_states and (any(
                    isinstance(goal, Goal) and goal.name == self.target_goal_name
                    for goal in self.model.grid.get_cell_list_contents(next_pos))):
                    done = True

                self._update_q_values(state, action, reward, next_state)

                total_return += reward

                if reward>=0:
                    pos = next_pos
                    state = next_state
            
            #self.epsilon = max(self.epsilon * 0.999, 0.01)
            self.epsilon = max(self.epsilon * 0.9999, 0.01)

        print(f"Episode {episode} finished in {training_step} steps with total return {total_return}.")

        self.save_q_values()

    def load_q_values(self, q_file):
        try:
            #print(f"Loading Q-values from {q_file}")
            self.q_values = np.load(q_file, allow_pickle=True).item()
            #print(f"Q-values from {q_file} have been loaded.")
        except FileNotFoundError:
            self.reset_q_values()
            print("File not found. Q-values have been reset.")

    def perform(self, pos, action) -> tuple:
        x = pos[0] + self.MOVEMENTS[action][0]
        y = pos[1] + self.MOVEMENTS[action][1]
        next_pos = (x, y)
        return next_pos

    def random_policy(self):
        return np.random.randint(self.NUM_OF_ACTIONS)

    def eps_greedy_policy(self, state):
        if self.training_step < self.MAX_NUM_TRAINING_STEPS or np.random.rand() < self.epsilon:
            self.training_step += 1
            return self.random_policy()
        else:
            q_values = [self.q_values[state, action] for action in range(self.NUM_OF_ACTIONS)]
            return np.argmax(q_values)
        
    def greedy_policy(self, state):
        q_values = [self.q_values[state, action] for action in range(self.NUM_OF_ACTIONS)]
        return np.argmax(q_values)

    def _update_q_values(self, state, action, reward, next_state):
        if action is not None:
            q_values = [self.q_values[next_state, action] for action in range(self.NUM_OF_ACTIONS)]
            max_q_value = np.max(q_values)
            self.q_values[state, action] += self.alpha * (
                    reward + self.gamma * max_q_value - self.q_values[state, action])
        
    def detect_oscillation(self):
        """
        Detecta si el bot está en un bucle de oscilación.
        Retorna True si se detecta oscilación, False en caso contrario.
        """
        if len(self.history) >= 4:
            if self.history[-1] == self.history[-3] and self.history[-2] == self.history[-4]:
                print("Oscilacion lineal")
                return True
            
        elif len(self.history) >= 8:
            if self.history[-1] == self.history[-5] and self.history[-2] == self.history[-6] and self.history[-3] == self.history[-7] and self.history[-4] == self.history[-8]:
                print("Oscilacion circular")
                return True
            
        return False
    
    def detect_oscillation_all_cases(self):
        """
        Detecta si el bot está en un bucle de oscilación.
        Retorna True si se detecta oscilación, False en caso contrario.
        """
        if self.target_goal_name == "":
            return False

        if len(self.history) >= 4:
            if (self.history[-1] == self.history[-3] and self.history[-2] == self.history[-4]):
                print("Oscilación lineal detectada (vertical/horizontal)")
                return True

        if len(self.history) >= 8:
            if (self.history[-1] == self.history[-5] and
                self.history[-2] == self.history[-6] and
                self.history[-3] == self.history[-7] and
                self.history[-4] == self.history[-8]):
                print("Oscilación circular detectada (cuadrado de 4 cuadros)")
                return True

        for pattern_length in range(4, len(self.history) // 2 + 1):
            if len(self.history) >= 2 * pattern_length:
                pattern1 = self.history[-pattern_length:]
                pattern2 = self.history[-2 * pattern_length:-pattern_length]
                if pattern1 == pattern2:
                    print(f"Oscilación circular detectada (patrón de {pattern_length} pasos)")
                    return True

        for length in range(2, len(self.history) // 2 + 1):
            if len(self.history) >= 2 * length:
                recent_pattern = self.history[-length:]
                previous_pattern = self.history[-2 * length:-length]
                if recent_pattern == previous_pattern:
                    print(f"Patrón repetitivo detectado con longitud {length}")
                    return True

        return False

        
    def backtrack(self, steps):
        """
        Retrocede una cantidad de pasos especificada.
        """
        if len(self.history) > steps:
            backtrack_pos = self.history[-(steps + 1)]
            self.model.grid.move_agent(self, backtrack_pos)
            self.history = self.history[:-steps]

    def find_random_empty_position(self):
        """ Encuentra una posición aleatoria vacía en la cuadrícula. """
        empty_positions = [
            (x, y) for x in range(self.model.grid.width)
            for y in range(self.model.grid.height)
            if self.model.grid.is_cell_empty((x, y))
        ]
        if empty_positions:
            return self.random.choice(empty_positions)
        else:
            raise ValueError("No hay posiciones vacías disponibles en la cuadrícula.")
        
    def deliver_or_charge(self):
        """
        Decide si es más viable para el bot entregar el paquete primero y luego cargar,
        o ir directamente a cargar según la batería y el costo energético estimado.
        """

        print(f"Cargador: {self.charger_name} asignado a bot {self.unique_id}")

        target_coords = next(((x, y) for (goal_id, x, y, name) in goals_collection if name == self.target_goal_name), None)
        charger_coords = next(((x, y) for (goal_id, x, y, name) in goals_collection if name == self.charger_name), None)

        if target_coords and charger_coords:
            cost_to_target = self.calculate_energy_cost(self.pos, target_coords, self.weight_box)
            cost_to_charger_from_target = self.calculate_energy_cost(target_coords, charger_coords, 0)  # Sin peso adicional después de dejar el paquete
            total_cost_delivery_first = cost_to_target + cost_to_charger_from_target

            cost_to_charger_direct = self.calculate_energy_cost(self.pos, charger_coords, self.weight_box)

            if total_cost_delivery_first < self.battery:
                print(f"Bot {self.unique_id} decide entregar el paquete primero y luego ir a cargar.")
                return
            elif cost_to_charger_direct < self.battery:
                print(f"Bot {self.unique_id} tiene batería baja ({self.battery}) y decide ir a '{self.charger_name}'.")
                self.aux_target = self.target_goal_name

            else:
                print(f"Bot {self.unique_id} no tiene suficiente batería para ninguna acción. Quedando en espera.")

    def calculate_energy_cost(self, start_coords, goal_coords, weight):
        """
        Calcula el costo de energía desde una posición específica a una meta basada en la distancia y el peso.
        """
        distance = abs(start_coords[0] - goal_coords[0]) + abs(start_coords[1] - goal_coords[1])
        energy_cost = distance * ((1 + weight * 0.1)/2) + 3
        return energy_cost



    def manhattan_heuristic(self, pos, goal):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    def a_star(self, start, goal):
        # Listas de nodos abiertos y cerrados
        open_list = []
        closed_list = set()

        # Añadir el nodo inicial a la lista abierta
        open_list.append((0 + self.manhattan_heuristic(start, goal), 0, start, None))

        while open_list:
            # Ordenar la lista abierta por f = g + h y seleccionar el nodo con menor f
            open_list.sort(key=lambda x: x[0])
            f, g, current, parent = open_list.pop(0)

            # Si alcanzamos el objetivo, reconstruimos el camino
            if current == goal:
                path = []
                while parent is not None:
                    path.append(current)
                    current = parent
                    parent = next((p for f, g, c, p in closed_list if c == current), None)
                path.reverse()
                return path

            # Añadir el nodo actual a la lista cerrada
            closed_list.add((f, g, current, parent))

            # Iterar sobre vecinos
            for neighbor in self.model.grid.iter_neighborhood(current, moore=False, include_center=False):
                if self.model.grid.is_cell_empty(neighbor) or neighbor == goal:
                    g_new = g + 1  # Distancia desde el inicio hasta el vecino
                    h_new = self.manhattan_heuristic(neighbor, goal)
                    f_new = g_new + h_new

                    # Si el vecino ya está en la lista cerrada, saltarlo
                    if any(neighbor == c for f, g, c, p in closed_list):
                        continue

                    # Si el vecino ya está en la lista abierta con un f mayor, actualizarlo
                    existing_node = next((i for i, (f, g, c, p) in enumerate(open_list) if c == neighbor), None)
                    if existing_node is not None:
                        if open_list[existing_node][0] > f_new:
                            open_list[existing_node] = (f_new, g_new, neighbor, current)
                    else:
                        open_list.append((f_new, g_new, neighbor, current))

        return []

    def a_star_team(self, start, goal):
        """
        Algoritmo A* para el líder del equipo.
        Verifica que las posiciones a la derecha, abajo derecha y abajo estén libres para el equipo.
        """
        # Listas de nodos abiertos y cerrados
        open_list = []
        closed_list = set()

        # Añadir el nodo inicial a la lista abierta
        open_list.append((0 + self.manhattan_heuristic(start, goal), 0, start, None))

        while open_list:
            # Ordenar la lista abierta por f = g + h y seleccionar el nodo con menor f
            open_list.sort(key=lambda x: x[0])
            f, g, current, parent = open_list.pop(0)

            # Si alcanzamos el objetivo, reconstruimos el camino
            if current == goal:
                path = []
                while parent is not None:
                    path.append(current)
                    current = parent
                    parent = next((p for f, g, c, p in closed_list if c == current), None)
                path.reverse()
                return path

            # Añadir el nodo actual a la lista cerrada
            closed_list.add((f, g, current, parent))

            # Iterar sobre vecinos
            for neighbor in self.model.grid.iter_neighborhood(current, moore=False, include_center=False):
                # Verificar si el líder y los otros tres espacios están libres
                if self.is_team_formation_valid(neighbor):
                    g_new = g + 1  # Distancia desde el inicio hasta el vecino
                    h_new = self.manhattan_heuristic(neighbor, goal)
                    f_new = g_new + h_new

                    # Si el vecino ya está en la lista cerrada, saltarlo
                    if any(neighbor == c for f, g, c, p in closed_list):
                        continue

                    # Si el vecino ya está en la lista abierta con un f mayor, actualizarlo
                    existing_node = next((i for i, (f, g, c, p) in enumerate(open_list) if c == neighbor), None)
                    if existing_node is not None:
                        if open_list[existing_node][0] > f_new:
                            open_list[existing_node] = (f_new, g_new, neighbor, current)
                    else:
                        open_list.append((f_new, g_new, neighbor, current))

        return []  # Si no se encuentra un camino, retornar una lista vacía


    def is_team_formation_valid(self, leader_pos):
        """
        Verifica si la formación del equipo es válida a partir de la posición del líder.
        El equipo sigue el patrón:
        - Bot 1 a la derecha
        - Bot 2 abajo
        - Bot 3 abajo a la derecha
        """
        # Posiciones donde deben estar los otros bots del equipo
        right_pos = (leader_pos[0] + 1, leader_pos[1])  # A la derecha
        down_pos = (leader_pos[0], leader_pos[1] - 1)  # Abajo
        down_right_pos = (leader_pos[0] + 1, leader_pos[1] - 1)  # Abajo a la derecha

        # Verificar que todas las posiciones están vacías o contienen a otro bot del equipo
        return (self.model.grid.is_cell_empty(right_pos) or self.is_team_member_at(right_pos)) and \
            (self.model.grid.is_cell_empty(down_pos) or self.is_team_member_at(down_pos)) and \
            (self.model.grid.is_cell_empty(down_right_pos) or self.is_team_member_at(down_right_pos))


    def is_team_member_at(self, pos):
        """
        Verifica si hay un miembro del equipo en una posición dada.
        """
        for agent in self.model.grid.get_cell_list_contents([pos]):
            if isinstance(agent, Bot) and agent.in_team_mode:
                return True
        return False

    def a_star_team_with_retries(self, start, goal, max_retries=4):
        """
        Algoritmo A* para el líder del equipo con intentos de ajuste. Si no se encuentra el camino, 
        reintenta con ajustes ligeros en la posición de inicio moviéndose una unidad en diferentes direcciones.
        """
        def find_path(start_pos):
            open_list = []
            closed_list = set()

            # Añadir el nodo inicial a la lista abierta
            open_list.append((0 + self.manhattan_heuristic(start_pos, goal), 0, start_pos, None))

            while open_list:
                # Ordenar la lista abierta por f = g + h y seleccionar el nodo con menor f
                open_list.sort(key=lambda x: x[0])
                f, g, current, parent = open_list.pop(0)

                # Si alcanzamos el objetivo, reconstruimos el camino
                if current == goal:
                    path = []
                    while parent is not None:
                        path.append(current)
                        current = parent
                        parent = next((p for f, g, c, p in closed_list if c == current), None)
                    path.reverse()
                    return path

                # Añadir el nodo actual a la lista cerrada
                closed_list.add((f, g, current, parent))

                # Explorar los vecinos
                for neighbor in self.model.grid.iter_neighborhood(current, moore=False, include_center=False):
                    if self.is_team_formation_valid(neighbor):
                        g_new = g + 1
                        h_new = self.manhattan_heuristic(neighbor, goal)
                        f_new = g_new + h_new

                        # Si el vecino ya está en la lista cerrada, saltarlo
                        if any(neighbor == c for f, g, c, p in closed_list):
                            continue

                        # Si el vecino ya está en la lista abierta con un f mayor, actualizarlo
                        existing_node = next((i for i, (f, g, c, p) in enumerate(open_list) if c == neighbor), None)
                        if existing_node is not None:
                            if open_list[existing_node][0] > f_new:
                                open_list[existing_node] = (f_new, g_new, neighbor, current)
                        else:
                            open_list.append((f_new, g_new, neighbor, current))

            return []

        # Intentar encontrar el camino con la posición original
        path = find_path(start)

        # Si no se encuentra camino, intentar con desplazamientos en diferentes direcciones
        if not path:
            print(f"Líder {self.unique_id} no encontró un camino directo. Intentando ajustes...")
            # Desplazamientos posibles: derecha, izquierda, arriba, abajo
            adjustments = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            retries = 0

            for adjustment in adjustments:
                retries += 1
                adjusted_start = (start[0] + adjustment[0], start[1] + adjustment[1])

                if self.model.grid.is_cell_empty(adjusted_start):  # Solo intentar si la celda ajustada está vacía
                    path = find_path(adjusted_start)
                    if path:
                        print(f"Líder {self.unique_id} encontró un camino ajustando en dirección {adjustment}.")
                        return path

                if retries >= max_retries:
                    break

        return path  # Si no encuentra camino, retorna el camino fallido o vacío





class Box(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)


class Goal(Agent):
    def __init__(self, unique_id, model, name):
        super().__init__(unique_id, model)
        self.name = name

class TaskManager:
    def __init__(self, environment):
        """
        Inicializa el TaskManager con una referencia al entorno.

        Parameters:
        - environment: Una instancia del modelo Environment.
        """
        self.environment = environment

    def assign_goal_to_bot(self, bot_id, goal_name):
        """
        Asigna un objetivo específico a un bot basado en el ID del bot y el nombre de la meta.

        Parameters:
        - bot_id: Identificador único del bot.
        - goal_name: Nombre de la meta a asignar al bot.
        """
        # Encontrar el bot con el ID especificado
        bot = next((agent for agent in self.environment.schedule.agents if isinstance(agent, Bot) and agent.unique_id == bot_id), None)

        # Buscar la meta con el nombre especificado en todas las celdas de la cuadrícula
        goal = None
        for cell in self.environment.grid.coord_iter():
            contents, pos = cell
            # Asegurarse de que contents es una lista de agentes
            cell_contents = self.environment.grid.get_cell_list_contents(pos)
            for agent in cell_contents:
                if isinstance(agent, Goal) and agent.name == goal_name:
                    goal = agent
                    break
            if goal:
                break

        if bot and goal:
            bot.target_goal_name = goal_name
            bot.rewards = {state: 1 if state == goal.pos else -1 for state in self.environment.states.values()}
            #print(f"Meta '{bot.target_goal_name}' asignada al bot {bot.unique_id}.")
        else:
            print(f"No se pudo asignar la meta. Verifique que el bot y la meta existan.")

    def monitor_bots_collision(self):
        """
        Monitorea a los bots para detectar colisiones en sus próximos pasos y previene que ocurra la colisión.
        """
        bots = [agent for agent in self.environment.schedule.agents if isinstance(agent, Bot)]
        next_positions = {}

        for bot in bots:
            if bot.action is not None:
                next_pos = bot.perform(bot.pos, bot.action)
                if next_pos in next_positions:
                    next_positions[next_pos].append(bot)
                else:
                    next_positions[next_pos] = [bot]

        for pos, bots_in_pos in next_positions.items():
            if len(bots_in_pos) > 1:
                print(f"Colisión detectada en posición {pos} entre los bots {', '.join(str(bot.unique_id) for bot in bots_in_pos)}")
                for bot in bots_in_pos:
                    bot.action = None  # Make bots wait

                # Negotiate movement: Allow the bot with the highest priority (e.g., unique_id) to move
                highest_priority_bot = min(bots_in_pos, key=lambda b: b.unique_id)
                highest_priority_bot.action = highest_priority_bot.greedy_policy(highest_priority_bot.state)
                #highest_priority_bot.action = highest_priority_bot.eps_greedy_policy(highest_priority_bot.state)
                print(f"Bot {highest_priority_bot.unique_id} moving to avoid collision.")

    def monitor_bots(self):
        """
        Monitorea a los bots para detectar colisiones en sus próximos pasos y previene que ocurra la colisión.
        """
        bots = [agent for agent in self.environment.schedule.agents if isinstance(agent, Bot)]
        next_positions = {}

        # Obtener las próximas posiciones de cada bot
        for bot in bots:
            if bot.action is not None:
                next_pos = bot.perform(bot.pos, bot.action)
                if next_pos in next_positions:
                    # Si otro bot ya se está moviendo a esta posición, hay una posible colisión
                    next_positions[next_pos].append(bot)
                else:
                    next_positions[next_pos] = [bot]

        # Revisar si hay colisiones y actuar en consecuencia
        for pos, bots_in_pos in next_positions.items():
            if len(bots_in_pos) > 1:
                # Si más de un bot se mueve a la misma posición, resolver el conflicto
                print(f"Colisión detectada en posición {pos} entre los bots {', '.join(str(bot.unique_id) for bot in bots_in_pos)}")

                # Ejemplo de resolución: Hacer que todos los bots excepto el primero en la lista esperen
                for i, bot in enumerate(bots_in_pos[1:], start=1):
                    print(f"Bot {bot.unique_id} esperando para evitar colisión")
                    bot.action = None  # No tomar acción (esperar)

    def get_free_bots_queue(self):
        """
        Obtiene una cola de todos los bots con isFree=True.

        Returns:
        - free_bots_queue: Cola de bots que están libres (isFree=True).
        """
        free_bots_queue = Queue()
        
        # Agregar bots libres a la cola
        for agent in self.environment.schedule.agents:
            if isinstance(agent, Bot) and agent.isFree:
                free_bots_queue.put(agent)

        return free_bots_queue
    
    def assign_tasks_to_free_bots(self):
        """
        Asigna tareas de la cola de artículos a los bots libres. Almacena la tarea completa en la variable 'task' del bot.
        """
        free_bots_queue = self.get_free_bots_queue()
        
        # Asignar artículos a bots libres desde la cola
        while not self.environment.articles_queue.empty() and not free_bots_queue.empty():
            article = self.environment.articles_queue.get()
            bot = free_bots_queue.get()

            bot.task = article
            bot.isFree = False
            bot.done = False
            print(f"Tarea {article} asignada al bot {bot.unique_id}")

    def assign_tasks_to_free_bots_extended_version(self):
        """
        Asigna tareas de la cola de artículos a los bots libres. 
        Si el artículo pesa más de 10 kg, asigna un equipo de bots.
        """
        free_bots_queue = self.get_free_bots_queue()
        
        # Asignar artículos a bots libres desde la cola
        while not self.environment.articles_queue.empty() and not free_bots_queue.empty():
            article = self.environment.articles_queue.get()
            article_id, weight, origin_name, destination_name = article

            # Si el peso es menor o igual a 10, asignar un solo bot
            if weight <= 10:
                bot = free_bots_queue.get()
                bot.task = article
                bot.isFree = False
                bot.done = False
                print(f"Tarea {article} asignada al bot {bot.unique_id}")

            # Si el peso está entre 10 y 20, asignar un equipo de 2 bots
            elif 10 < weight <= 20:
                team_size = 2
                team_members = self.assign_team_to_article(team_size, article, free_bots_queue)
                if team_members:
                    print(f"Equipo de {team_size} bots asignado a tarea {article_id} ({weight} kg).")
                else:
                    print(f"No hay suficientes bots libres para la tarea {article_id}. Reinsertando en la cola.")
                    self.environment.articles_queue.put(article)

            # Si el peso es mayor a 20, asignar un equipo de 4 bots
            elif weight > 20:
                team_size = 4
                team_members = self.assign_team_to_article(team_size, article, free_bots_queue)
                if team_members:
                    print(f"Equipo de {team_size} bots asignado a tarea {article_id} ({weight} kg).")
                else:
                    print(f"No hay suficientes bots libres para la tarea {article_id}. Reinsertando en la cola.")
                    self.environment.articles_queue.put(article)

    def assign_team_to_article(self, team_size, article, free_bots_queue):
        """
        Asigna un equipo de bots a una tarea de equipo.
        Si no hay suficientes bots libres, los reingresa en la cola y retorna None.
        """
        team_members = []

        # Intentar obtener un equipo completo
        for _ in range(team_size):
            if not free_bots_queue.empty():
                bot = free_bots_queue.get()
                team_members.append(bot)
            else:
                # No hay suficientes bots, devolver los que ya hemos sacado a la cola
                for bot in team_members:
                    free_bots_queue.put(bot)
                return None

        self.environment.bot_teams.append(team_members)

        # Si se obtiene un equipo completo, asignar la tarea a todos los bots del equipo
        for bot in team_members:
            bot.task = article
            bot.isFree = False
            bot.done = False
            bot.in_team_mode = True
            bot.team_size = team_size

        return team_members


    def manage_bot_movements(self):
        """ Gestiona y coordina los movimientos de los bots para evitar colisiones. """
        planned_positions = {}

        # Iterar sobre todos los bots en el entorno
        for bot in self.environment.schedule.agents:
            if isinstance(bot, Bot):

                if bot.state is None:
                    bot.state = self.environment.states[bot.pos]
                    
                # Calcular la próxima posición basada en su acción planificada
                next_pos = bot.perform(bot.pos, bot.greedy_policy(bot.state))
                
                # Si la posición ya está en uso, se detecta una colisión potencial
                if next_pos in planned_positions:
                    planned_positions[next_pos].append(bot)
                else:
                    planned_positions[next_pos] = [bot]

        # Ajustar movimientos para evitar colisiones
        for pos, bots_in_pos in planned_positions.items():
            if len(bots_in_pos) > 1:
                # Hay más de un bot planeando moverse a la misma posición

                for bot in bots_in_pos:
                    # Replanificar la acción para cada bot involucrado
                    bot.action = self.find_alternative_action(bot)
                    bot.next_pos = bot.perform(bot.pos, bot.action)
                    #print(f"Bot {bot.unique_id} replanificado para moverse a {bot.next_pos}")

    def find_alternative_action(self, bot):
        """ Encuentra una acción alternativa para el bot para evitar colisiones. """
        possible_actions = list(range(bot.NUM_OF_ACTIONS))
        random.shuffle(possible_actions)  # Aleatorizar las acciones

        for action in possible_actions:
            alternative_pos = bot.perform(bot.pos, action)
            # Verificar si la nueva posición alternativa está libre y no es una zona de colisión
            if self.environment.grid.is_cell_empty(alternative_pos) and not self.is_collision_zone(alternative_pos):
                return action

        # Si no hay alternativas seguras, el bot espera (puede ajustarse según la lógica deseada)
        return None

    def is_collision_zone(self, pos):
        """ Verifica si la casilla es una zona de colisión potencial. """
        for agent in self.environment.schedule.agents:
            if isinstance(agent, Bot) and agent.pos == pos:
                return True
        return False

    def manage_team_movements(self):
        """
        Gestiona los movimientos de los equipos de bots. El líder del equipo toma las decisiones
        y el resto de los miembros copian sus movimientos.
        """
        for team in self.environment.bot_teams:
            if team:
                leader = team[0]  # El primer bot del equipo es el líder
                self.synchronize_team_with_leader(leader, team)

    def synchronize_team_with_leader(self, leader, team):
        """
        Hace que todos los miembros del equipo sigan los movimientos del líder.
        """
        # Si el líder tiene una acción asignada, el resto del equipo debe copiarla
        if leader.target_name_goal != "":
            if leader.action is not None:
                for bot in team[1:]:  # Excluir al líder (primer bot)
                    bot.action = leader.action
                    bot.next_pos = leader.next_pos
                    bot.next_state = leader.next_state
            else:
                print(f"El líder del equipo {leader.unique_id} no tiene una acción válida.")
