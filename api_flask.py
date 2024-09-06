from flask import Flask, jsonify
from flask_cors import CORS
import subprocess
import os
import signal
import json

app = Flask(__name__)
CORS(app)

server_process = None

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
    
@app.route('/get-points', methods=['GET'])
def get_points():
    try:
        # Abre y lee el archivo JSON local
        with open('robot_data.json', 'r') as json_file:
            data = json.load(json_file)  # Carga el contenido del archivo en la variable 'data'
        
        # Devuelve el contenido del archivo JSON
        return jsonify(data), 200
    except Exception as e:
        # Si ocurre algún error, lo capturamos y devolvemos un mensaje de error
        return jsonify({"status": "Error", "message": str(e)}), 500
    
@app.route('/get-summary', methods=['GET'])
def get_points():
    try:
        # Abre y lee el archivo JSON local
        with open('"simulation_summary.json"', 'r') as json_file:
            data = json.load(json_file)  # Carga el contenido del archivo en la variable 'data'
        
        # Devuelve el contenido del archivo JSON
        return jsonify(data), 200
    except Exception as e:
        # Si ocurre algún error, lo capturamos y devolvemos un mensaje de error
        return jsonify({"status": "Error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
