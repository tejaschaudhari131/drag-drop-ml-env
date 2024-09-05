# Drag-and-Drop Machine Learning Environment

This project is a Scratch-like tool designed to help users implement machine learning pipelines using a drag-and-drop interface. It includes built-in tutorials for each concept, making it an educational tool for those new to machine learning. Additionally, the project now includes the ability to export trained models in multiple formats (`pickle`, `tensorflow`, `onnx`) and serves models via an API for real-time predictions.

## Features

### Drag-n-Drop Interface:
- **Intuitive interface** for building machine learning models.
- Components for **data preprocessing**, **model selection**, **training**, and **evaluation**.
- **Live interaction** with the workspace, allowing users to visually design pipelines.

### Model Export:
- **Export trained models** for use in other applications.
- **Supported formats** include:
  - **Pickle**: For traditional Scikit-learn models.
  - **TensorFlow (Keras)**: For deep learning models.
  - **ONNX**: For interoperability with multiple platforms, including ONNX Runtime.

### Model Serving via API:
- **Real-time model serving**: Expose trained models as REST APIs.
- **Predict using external applications** by sending data to the API.
- Supports multiple formats for prediction: **ONNX**, **TensorFlow Lite**, and **Scikit-learn**.

### Built-in Tutorials:
- **Step-by-step guides** integrated into the interface.
- Tutorials cover **basic to advanced machine learning concepts** to help users get started.

### Evaluation and Visualization:
- **Evaluate models** with various metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.
- **Confusion matrix and ROC curve** visualizations to help interpret model performance.
- **Feature importance plots** for models that support it (e.g., tree-based models).

## Technology Stack

- **Backend**: Python (Flask, Scikit-learn, TensorFlow, ONNX)
- **Frontend**: JavaScript, HTML, CSS
- **Visualization**: D3.js and Matplotlib for data visualization
- **Model Serving**: ONNX Runtime, TensorFlow Lite, Flask REST API

## Installation

### 1. Clone the repository:
```bash
git clone https://github.com/tejaschaudhari131/drag-drop-ml-env.git
cd drag-drop-ml-env
```

### 2. Set up the environment:
You need to install the required Python dependencies by running:
```bash
pip install -r requirements.txt
```

### 3. Run the application:
Start the Flask server:
```bash
python app.py
```

### 4. Access the interface:
Open your browser and go to `http://localhost:5000` to access the drag-and-drop interface.

## Usage

### 1. Drag-and-Drop Interface
- Drag components from the toolbox (Preprocessing, Models, etc.) onto the workspace.
- Build your machine learning pipeline by chaining together different components.
- Train and evaluate your models directly within the interface.

### 2. Model Export
After training a model, you can export it in the following formats:
- **Pickle**: Traditional serialization for Scikit-learn models.
- **TensorFlow**: Export Keras models as HDF5 (`.h5`) files.
- **ONNX**: Interoperable format that can be used across different platforms.

The exported models will be saved in the `models/` directory.

### 3. Model Serving via API
Once a model is trained and exported, it can be served via the API for real-time predictions.

#### Example: Making Predictions with an Exported Model

You can use `curl` or any HTTP client to send data to the API and get predictions from a trained model.

##### API Endpoint: `POST /predict/<model_name>`

- **URL**: `http://localhost:5000/predict/<model_name>`
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Body**:
  ```json
  {
    "inputs": [5.1, 3.5, 1.4, 0.2]  # Example feature input
  }
  ```

#### Example Request (using `curl`):
```bash
curl -X POST http://localhost:5000/predict/random_forest \
  -H "Content-Type: application/json" \
  -d '{"inputs": [5.1, 3.5, 1.4, 0.2]}'
```

#### Example Response:
```json
{
  "prediction": [0]  # Example class prediction from the model
}
```

### 4. Model Evaluation
- **Confusion Matrix**: Visualizes the performance of classification models.
- **ROC Curve**: Displays the trade-off between true positive and false positive rates.
- **Classification Report**: Provides detailed metrics such as precision, recall, and F1-score for each class.

## Supported Models

The following models are available for training:

1. **Logistic Regression**
2. **Decision Tree**
3. **Random Forest**
4. **Gradient Boosting**
5. **Support Vector Machines (SVM)**
6. **K-Nearest Neighbors (KNN)**
7. **Naive Bayes**
8. **XGBoost**
9. **Neural Networks (TensorFlow/Keras)**
10. **KMeans Clustering**

## Export Formats

The following formats are supported for exporting models:

- **Pickle**: Use for traditional machine learning models (e.g., Scikit-learn).
- **TensorFlow**: Save deep learning models in HDF5 format for use in TensorFlow applications.
- **ONNX**: Export models for use across different platforms that support ONNX (e.g., PyTorch, ONNX Runtime).

## API Endpoints

### 1. **Upload Dataset**
- **URL**: `POST /upload`
- **Description**: Upload a dataset to be used in training.

### 2. **Preprocess Dataset**
- **URL**: `POST /preprocess`
- **Description**: Apply preprocessing steps to the uploaded dataset.

### 3. **Train Model**
- **URL**: `POST /train_model`
- **Description**: Train a machine learning model with selected parameters.

### 4. **Export Model**
- **URL**: `POST /export_model`
- **Description**: Export a trained model in a specified format.

### 5. **Model Prediction**
- **URL**: `POST /predict/<model_name>`
- **Description**: Make predictions using a trained and exported model.

### 6. **List Saved Models**
- **URL**: `GET /models`
- **Description**: Get a list of all models saved in the `models/` directory.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests to improve the functionality or fix bugs.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any inquiries, please contact:

**Tejaram Chaudhari**: tejaschaudhari131@gmail.com
