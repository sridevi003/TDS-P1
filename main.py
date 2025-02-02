from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/run', methods=['POST'])
def run_task():
    task = request.args.get('task', '')
    if task:
        return jsonify({'message': f'Task received: {task}'}), 200
    else:
        return jsonify({'error': 'No task provided'}), 400

@app.route('/read', methods=['GET'])
def read_file():
    path = request.args.get('path', '')
    if path:
        try:
            with open(path, 'r') as file:
                content = file.read()
            return content, 200
        except FileNotFoundError:
            return jsonify({'error': 'File not found'}), 404
    else:
        return jsonify({'error': 'No path provided'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
