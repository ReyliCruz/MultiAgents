from mesa.agent import Agent
import numpy as np


class Bot(Agent):

    MAX_NUM_TRAINING_STEPS = 1000
    NUM_OF_ACTIONS = 4

    # Define the movements (0: down, 1: right, 2: up, 3: left)
    MOVEMENTS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def __init__(self, unique_id, model, q_file=None):
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
        self.rewards = {}
        self.visible_goal = True

        self.epsilon = 0.1
        self.alpha = 0.1
        self.gamma = 0.9

        self.num_states = self.model.grid.width * self.model.grid.height

        if q_file is None:
            self.reset_q_values()
        else:
            self.load_q_values(q_file)

    def reset_q_values(self):
        self.q_values = {(state, action): np.random.uniform(0, .01)
                         for state in range(self.num_states)
                         for action in range(self.NUM_OF_ACTIONS)}

    def step(self) -> None:
        if self.state is None:
            self.state = self.model.states[self.pos]

        # Agent chooses an action from the policy
        self.action = self.eps_greedy_policy(self.state)

        # Get the next position
        self.next_pos = self.perform(self.pos, self.action)
        self.next_state = self.model.states[self.next_pos]

    def advance(self) -> None:

        if self.target_goal_name == "":
            print("Sin meta asignada")
            return
        
        # Check if the agent can move to the next position
        if self.model.grid.is_cell_empty(self.next_pos) or (
            self.next_state in self.model.goal_states and 
            any(isinstance(goal, Goal) and goal.name == self.target_goal_name 
                for goal in self.model.grid.get_cell_list_contents(self.next_pos))
        ):
            if self.next_state in self.model.goal_states:
                # Remove the goal agent from the grid
                self.model.grid.remove_agent(self.model.grid.get_cell_list_contents(self.next_pos)[0])
                self.done = True

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
        inital_pos = self.pos
        initial_state = self.model.states[inital_pos]

        for episode in range(1000):
            training_step = 0
            done = False
            total_return = 0
            movements = 0
            pos = inital_pos
            state = initial_state

            while not done:
                training_step += 1
                action = self.eps_greedy_policy(state)

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
            
            
            self.epsilon = max(self.epsilon * 0.99, 0.01)

        print(f"Episode {episode} finished in {training_step} steps with total return {total_return}.")

        self.save_q_values()


    def load_q_values(self, q_file):
        try:
            print(f"Loading Q-values from {q_file}")
            self.q_values = np.load(q_file, allow_pickle=True).item()
            print(f"Q-values from {q_file} have been loaded.")
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

    def _update_q_values(self, state, action, reward, next_state):
        q_values = [self.q_values[next_state, action] for action in range(self.NUM_OF_ACTIONS)]
        max_q_value = np.max(q_values)
        self.q_values[state, action] += self.alpha * (
                reward + self.gamma * max_q_value - self.q_values[state, action])


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
            # Asignar la meta al bot
            bot.target_goal_name = goal_name
            bot.rewards = {state: 1 if state == goal.pos else -1 for state in self.environment.states.values()}
            print(f"Meta '{bot.target_goal_name}' asignada al bot {bot.unique_id}.")
        else:
            print(f"No se pudo asignar la meta. Verifique que el bot y la meta existan.")

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
