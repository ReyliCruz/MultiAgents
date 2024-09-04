from flask import Flask, jsonify
from flask_cors import CORS
import subprocess
import os
import signal

app = Flask(__name__)
CORS(app)

server_process = None

@app.route('/get-points', methods=['GET'])
def get_points():
    try:
        data = {
            "robots": [
                {
                    "spawnPosition": {
                        "x": 0,
                        "y": 0
                    },
                    "path": [
                        {
                        "x": 1,
                        "y": 9
                        },
                        {
                        "x": 1,
                        "y": 2
                        },
                        {
                        "x": 7,
                        "y": 2
                        },
                        {
                        "x": 7,
                        "y": 9
                        },
                        {
                        "x": 9,
                        "y": 1
                        },
                        {
                        "x": 9,
                        "y": 5
                        },
                        {
                        "x": 5,
                        "y": 6
                        },
                        {
                        "x": 6,
                        "y": 6
                        },
                        {
                        "x": 6,
                        "y": 7
                        }
                    ]
                },

                {
                    "spawnPosition": {
                        "x": 1,
                        "y": 1
                    },
                    "path": [
                        {
                        "x": 2,
                        "y": 10
                        },
                        {
                        "x": 2,
                        "y": 3
                        },
                        {
                        "x": 8,
                        "y": 3
                        },
                        {
                        "x": 8,
                        "y": 10
                        },
                        {
                        "x": 10,
                        "y": 2
                        },
                        {
                        "x": 10,
                        "y": 6
                        },
                        {
                        "x": 6,
                        "y": 7
                        },
                        {
                        "x": 7,
                        "y": 7
                        },
                        {
                        "x": 7,
                        "y": 8
                        }
                    ]
                }
            ]
        }
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"status": "Error", "message": str(e)}), 500


@app.route('/run-simulation', methods=['GET'])
def run_simulation():
    global server_process

    if server_process is None or server_process.poll() is not None:
        try:
            server_process = subprocess.Popen(['python', 'server.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return jsonify({"status": "Simulación iniciada"}), 200
        except Exception as e:
            return jsonify({"status": "Error", "message": str(e)}), 500
    else:
        return jsonify({"status": "El servidor ya está en ejecución"}), 200

@app.route('/stop-simulation', methods=['GET'])
def stop_simulation():
    global server_process

    if server_process is not None:
        os.kill(server_process.pid, signal.SIGTERM)
        server_process.wait()
        server_process = None
        return jsonify({"status": "Simulación detenida"}), 200
    else:
        return jsonify({"status": "No hay simulación en ejecución"}), 400

@app.route('/step-simulation', methods=['GET'])
def step_simulation():
    global server_process

    if server_process is not None and server_process.poll() is None:
        try:
            subprocess.Popen(['python', '-c', 'from server import advance_simulation_step; advance_simulation_step()'])
            return jsonify({"status": "Step ejecutado"}), 200
        except Exception as e:
            return jsonify({"status": "Error al ejecutar el step", "message": str(e)}), 500
    else:
        return jsonify({"status": "El servidor no está en ejecución"}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
