from flask import Flask, render_template, request, jsonify
import os
from ml_components import preprocess, models, evaluation, export_model
from werkzeug.utils import secure_filename
import json

app = Flask(__name__)

# Set up file upload configurations
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

# Utility function to check if file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route to render the main interface
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle dataset upload
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        # Further processing of the file can be added here
        return jsonify({'success': True, 'file_path': file_path})
    return jsonify({'error': 'File type not allowed'})

# Preprocessing route
@app.route('/preprocess', methods=['POST'])
def preprocess_data():
    data = request.json
    processed_data = preprocess.handle_preprocessing(data['dataset'], data['preprocessing_steps'])
    return jsonify(processed_data)

# Route to handle model training
@app.route('/train_model', methods=['POST'])
def train_model():
    data = request.json
    X_train, X_test, y_train, y_test = preprocess.data_split(data['dataset'])
    
    trained_model = models.train_model(data['model_type'], X_train, y_train)
    evaluation_results = evaluation.evaluate_model(trained_model, X_test, y_test)
    
    return jsonify(evaluation_results)

# Route to export trained model in different formats
@app.route('/export_model', methods=['POST'])
def export_model_route():
    model_type = request.json['model_type']
    format_type = request.json['format']
    filepath = export_model.export_trained_model(model_type, format_type)
    return jsonify({"model_path": filepath})

# Route to fetch available tutorials
@app.route('/tutorials', methods=['GET'])
def tutorials():
    with open('tutorials/tutorials.json') as f:
        tutorials_data = json.load(f)
    return jsonify(tutorials_data)

# Helper route for progress updates
@app.route('/progress', methods=['GET'])
def progress():
    # Simulate returning progress of the training process (e.g., 0-100%)
    progress = 50  # Placeholder value, logic for real-time progress would be more complex
    return jsonify({'progress': progress})

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
