from mesa.agent import Agent
import numpy as np


class Bot(Agent):

    MAX_NUM_TRAINING_STEPS = 1000
    NUM_OF_ACTIONS = 4

    # Define the movements (0: down, 1: right, 2: up, 3: left)
    MOVEMENTS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def __init__(self, unique_id, model, q_file=None, target_goal_name=None):
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
        self.target_goal_name = target_goal_name

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
        np.save(f"./q_values{self.unique_id}.npy", self.q_values)

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

                reward = self.model.rewards[next_state]

                if next_state in self.model.goal_states:
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