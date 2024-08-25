import gymnasium as gym
import random
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display import display, clear_output
from time import sleep

def mostrar(env, wait=0):
    """
    Muestra el entorno a modo de animación

    :param env: Entorno de OpenAI Gym - FrozenLake
    :param wait: Tiempo de espera en segundo para realizar el siguiente paso
    :return: None
    """
    screen = env.render(mode="rgb_array")
    plt.imshow(screen)
    plt.axis(False)

    sleep(wait)
    display(plt.gcf())
    clear_output(True)



def ejecutar_juego(env, policy, num_iterations=100, wait=0):
    """
    Corre la simulación de un episodio siguiendo una política dada.

    :param env: Entorno de OpenAI Gym - FrozenLake
    :param policy: Política para seleccionar la acción, esta puede ser una función, un diccionario de estados
        o uno de estado-acción
    :param num_iterations: El número de pasos a correr.
    :param wait: Tiempo de espera en segundo para presentar la visualización
    :return: None
    """
    state = env.reset()
    for i in range(num_iterations):

        if hasattr(policy, "__call__"):
            action = policy(state)
        else:
            current_policy = policy[state]
            if isinstance(current_policy, dict):
                probabilities = np.array([current_policy[action] for action in range(env.action_space.n)])
                action = random.choices(range(env.action_space.n), probabilities / np.sum(probabilities))[0]
            else:
                action = policy[state]
        state, _, done, _ = env.step(action)
        mostrar(env, wait)
        if done:
            break



def epsilon_greedy_policy(env, Q, state, epsilon=0.1):
    # Select action with Epsilon-Greedy Algorithm
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        Q_list = [Q[(state, act)] for act in range(env.action_space.n)]
        return random.choice([ind for ind, x in enumerate(Q_list) if x == max(Q_list)])
    


def q_learning(env, gamma=0.9, epsilon=0.1, alpha=0.618, num_episodes=1000, Q=None):
    if Q is None:
        Q = {(state, action): 0
             for state in range(env.observation_space.n)
             for action in range(env.action_space.n)}
    G_history = []
    G_prom_hist = []

    for ep_count in range(num_episodes):
        done = False
        G = 0
        step = 1

        # Generate a trajectory
        state = env.reset()
        while not done:
            step += 1

            action = epsilon_greedy_policy(env, Q, state, epsilon)

            next_state, reward, done, _ = env.step(action)

            Q_list_next_state = [Q[(next_state, act)] for act in range(env.action_space.n)]

            # Update rule
            Q[(state, action)] += alpha * (reward + gamma * np.max(Q_list_next_state) - Q[(state, action)])

            G += reward
            state = next_state

        G_history.append(G)
        G_prom_hist.append(G / step)

        if ep_count % (num_episodes // 10) == 0:
            print(f"Episodio {ep_count}, Recompensa total: {G}",
                  f"Recompensa promedio: {np.array(G_history).mean()} ")

    env.close()

    return Q, G_history, G_prom_hist



mapa = [
    'SFFFF',
    'HHFHH',
    'FFFFF',
    'FHHHF',
    'FFGFF'
]

# Definimos el entorno para la simulación
np.random.seed(123)
entorno = gym.make('FrozenLake-v1', desc=mapa, is_slippery=True)
entorno.reset()

# Establecemos los parámetros
parameters = dict(gamma=1, alpha=0.5, epsilon=0.01, num_episodes=5000)

Q_qlearn, G_history_qlearn, G_prom_qlearn = q_learning(entorno, **parameters)

politica_eps_greedy = lambda x: epsilon_greedy_policy(
    entorno, Q_qlearn, x, epsilon=0.1)

entorno = gym.make('FrozenLake-v1', desc=mapa, is_slippery=True)
ejecutar_juego(entorno, politica_eps_greedy, 100)