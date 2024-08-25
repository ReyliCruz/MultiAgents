import heapq
import gymnasium as gym
import numpy as np

from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import json

# Define the port number to listen on
PORT = 6969

# Get the environment
DESC = [
    'BBBBBBBBBBBBBBBBBBBB',
    'BSFFFFFFFFFFFFFFFBBF',
    'BBFFFFFBBBBBBFFFFBFF',
    'BFFBBBBBFFFFBFFBBBFB',
    'BFFFFBBBFBBBBFFFBFFB',
    'BBBFBBFFFBBFFFFBBFFB',
    'BFFBBFFBFFFFBBFFBBFB',
    'BBFBFFFFBBBFFBBFFFFB',
    'BFBBFFFFFBFFBBFFFFFB',
    'BFFFFFFBBFFFFFBBBBBB',
    'BFFFFBBFFBBBBBBFFFFB',
    'BFBBFFFFFBBFFFFBBBFB',
    'BFFFFBBFFFFBBFFBBFFB',
    'FFBBBBFFFFFBFFFFFBFB',
    'BFBFBBFFFFFFFBBFFFFB',
    'BFBBFFBFFFFFBBFBBBBB',
    'BFFFFFBBFFFFBBFFFFFB',
    'FBFBFBFBBFFFFFFBBBFB',
    'BFFFFFFFBBFGFFFFFFBB',
    'BBBBBBBBBBBBBBBBBBBB'
]


def from_desc_to_maze(desc):
    """ In the FrozenLake environment, the grid is described by a list of strings, where:
    - 'S' is the starting point of the agent,
    - 'F' is the regular frozen surface or walkable path,
    - 'H' is the hole or ending state,
    - 'G' is the goal.
    """
    start = None
    end = None

    desc_ = [list(row) for row in desc]
    desc_matrix = np.array(desc_)

    maze = np.zeros(desc_matrix.shape, dtype=int)

    for i in range(desc_matrix.shape[0]):
        for j in range(desc_matrix.shape[1]):
            if desc_matrix[i, j] == 'S':
                start = (i, j)
            elif desc_matrix[i, j] == 'G':
                end = (i, j)
            elif desc_matrix[i, j] == 'H':
                maze[i, j] = 1
    return maze, start, end


# A* algorithm implementation
def a_star(desc):
    """ A* algorithm: returns the path from start to end in the maze (environment)
    """

    # Get the maze, start, and end positions
    maze, start, end = from_desc_to_maze(desc)

    # Define the movements (down, up, right, left)
    movements = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    # Initialize the open list as a heap and the closed list
    open_list = []  # f-score, position
    heapq.heappush(open_list, (0, start))  # f-score, position
    closed_list = set()
    came_from = {}

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    g_score = {start: 0}

    # A* algorithm
    while open_list:
        # Get the current node with the lowest f-score
        _, current = heapq.heappop(open_list)

        # Check if the current node is the goal
        if current == end:
            # Reconstruct the path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        # Add the current node to the closed list
        closed_list.add(current)

        # Check the neighbors, only the ones that are not obstacles
        for movement in movements:
            # Get the neighbor position based on the movements and the current position
            neighbor = (current[0] + movement[0], current[1] + movement[1])

            # Check if the neighbor is within bounds
            if not (0 <= neighbor[0] < maze.shape[0] and 0 <= neighbor[1] < maze.shape[1]):
                continue

            if maze[neighbor[0], neighbor[1]] == 1 or neighbor in closed_list:
                continue

            # Calculate the tentative g-score
            tentative_g = g_score[current] + 1

            # Check if the neighbor is not in the open list or the tentative g-score is lower than the previous one
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, end)
                heapq.heappush(open_list, (f_score, neighbor))

    return []


def from_positions_to_json(positions):
    # Spawn positions
    spawn_position = positions[0]

    pos_list = []
    for pos in positions:
        pos_list.append({"x": pos[0], "y": pos[1]})

    json_data = {
        "spawnX": spawn_position[0],
        "spawnY": spawn_position[1],
        "path": pos_list
    }

    return json_data


# Transform the list of positions into a list of actions
def from_positions_to_actions(positions):
    actions = []
    for i in range(1, len(positions)):
        movement = (positions[i][0] - positions[i - 1][0], positions[i][1] - positions[i - 1][1])
        if movement == (0, 1):
            actions.append(2)  # right
        elif movement == (1, 0):
            actions.append(1)  # down
        elif movement == (0, -1):
            actions.append(0)  # left
        elif movement == (-1, 0):
            actions.append(3)  # up
    return actions


class Server(BaseHTTPRequestHandler):

    def _set_response(self, content_type="text/html", status_code=200):
        """
        Helper method to set the HTTP response status and headers.
        """
        self.send_response(status_code)
        self.send_header('Content-type', content_type)
        self.end_headers()

    def do_GET(self):
        """
        Handles GET requests by logging the request and sending a basic HTML response.
        """
        logging.info("GET request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers))
        self._set_response()
        response = {
            "message": f"GET request for {self.path}",
            "status": "success"
        }
        self.wfile.write(json.dumps(response).encode('utf-8'))

    def do_POST(self):
        """
        Handles POST requests by logging the request, processing JSON data,
        and responding with a JSON message.
        """
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)

        try:
            # Parse JSON data
            json_data = json.loads(post_data)
            logging.info("POST request,\nPath: %s\nHeaders:\n%s\n\nBody:\n%s\n",
                         str(self.path), str(self.headers), json.dumps(json_data, indent=4))

            # Example giving the updated positions of the boids
            if json_data.get("action") == "next_step":
                # Update the positions of the boids
                positions = next(positions_from_sim)
                response_data = positions_to_json(positions)
            else:
                # Process the received JSON data
                response_data = self.process_data(json_data)

            # Set response and send JSON
            self._set_response(content_type="application/json")
            response = {
                "status": "success",
                "data": response_data
            }
        except json.JSONDecodeError:
            # Handle invalid JSON input
            self._set_response(content_type="application/json", status_code=400)
            response = {
                "status": "error",
                "message": "Invalid JSON data received"
            }

        # Send the response as JSON
        self.wfile.write(json.dumps(response).encode('utf-8'))

    def process_data(self, data):
        """
        Example method to process the received JSON data. Modify as needed.
        """
        try:
            x = data.get('x', 0)
            y = data.get('y', 0)
            z = data.get('z', 0)
            return {"x": x, "y": y, "z": z}
        except KeyError as e:
            logging.error(f"Missing key in input data: {e}")
            return {"error": f"Missing key: {e}"}


def positions_to_json(positions):
    """
    Converts a list of positions to a JSON string.
    """
    return json.dumps(positions)


def run(server_class=HTTPServer, handler_class=Server, port=PORT):
    logging.basicConfig(level=logging.INFO)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    logging.info("Starting httpd...\n")  # HTTPD is HTTP Daemon!
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:  # CTRL+C stops the server
        pass
    httpd.server_close()
    logging.info("Stopping httpd...\n")


# Define the main method to run from the command line
if __name__ == '__main__':
    # Replace 'B' with 'H' in desc
    desc = [row.replace('B', 'H') for row in DESC]

    # Create the environment
    env = gym.make("FrozenLake-v1", desc=desc, is_slippery=False, render_mode="human")
    observation, info = env.reset()

    # Get the path from start to end
    steps = a_star(desc)
    print("Path:", steps)

    # Get the list of actions
    actions_list = from_positions_to_actions(steps)
    print("Actions:", actions_list)

    for action in actions_list:
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            break

    # Convert the list of positions into a JSON object
    data = from_positions_to_json(steps)
    print(data)

    # Transform data to a generator object
    positions_from_sim = (d for d in data["path"])

    # Start the server
    run()
