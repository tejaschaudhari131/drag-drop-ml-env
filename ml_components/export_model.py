import os
import pickle
import tensorflow as tf
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx
import onnxruntime as ort

MODELS_FOLDER = 'models'  # Folder where models are saved

# Ensure the models directory exists
if not os.path.exists(MODELS_FOLDER):
    os.makedirs(MODELS_FOLDER)


def export_model(model, model_name, format_type='pickle'):
    """
    Save the trained model in the specified format (Pickle, TensorFlow, or ONNX).
    
    Args:
    - model: Trained model to save
    - model_name: Name to use for the saved model file
    - format_type: Format in which to save the model ('pickle', 'tensorflow', 'onnx')
    
    Returns:
    - model_path: Path to the saved model file
    """
    if format_type == 'pickle':
        return save_pickle_model(model, model_name)
    elif format_type == 'tensorflow':
        return save_tensorflow_model(model, model_name)
    elif format_type == 'onnx':
        return save_onnx_model(model, model_name)
    else:
        raise ValueError(f"Unsupported format type '{format_type}'. Supported types: 'pickle', 'tensorflow', 'onnx'.")


def save_pickle_model(model, model_name):
    """
    Save the model using Pickle serialization.
    
    Args:
    - model: Trained model to save
    - model_name: Name to use for the saved model file
    
    Returns:
    - model_path: Path to the saved model file
    """
    model_file = os.path.join(MODELS_FOLDER, f'{model_name}.pkl')
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    return model_file


def load_pickle_model(model_name):
    """
    Load a model saved as a Pickle file.
    
    Args:
    - model_name: Name of the model to load
    
    Returns:
    - model: Loaded model object
    """
    model_file = os.path.join(MODELS_FOLDER, f'{model_name}.pkl')
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"No model found with the name '{model_name}' in Pickle format.")
    
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    return model


def save_tensorflow_model(model, model_name):
    """
    Save a TensorFlow (Keras) model.
    
    Args:
    - model: Trained Keras model to save
    - model_name: Name to use for the saved model file
    
    Returns:
    - model_path: Path to the saved model file
    """
    model_file = os.path.join(MODELS_FOLDER, f'{model_name}.h5')
    model.save(model_file)  # Save model in HDF5 format
    return model_file


def load_tensorflow_model(model_name):
    """
    Load a TensorFlow (Keras) model from a file.
    
    Args:
    - model_name: Name of the model to load
    
    Returns:
    - model: Loaded Keras model
    """
    model_file = os.path.join(MODELS_FOLDER, f'{model_name}.h5')
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"No model found with the name '{model_name}' in TensorFlow format.")
    
    return tf.keras.models.load_model(model_file)


def save_onnx_model(model, model_name):
    """
    Convert and save a Scikit-learn model to ONNX format.
    
    Args:
    - model: Trained Scikit-learn model to convert
    - model_name: Name to use for the saved model file
    
    Returns:
    - model_path: Path to the saved ONNX model file
    """
    # Check if model is compatible with ONNX conversion
    if not hasattr(model, 'predict'):
        raise ValueError("The model must be a Scikit-learn model or a model that supports ONNX conversion.")
    
    # Prepare the initial type for the ONNX model (assuming the input is 2D floats)
    initial_type = [('float_input', FloatTensorType([None, model.n_features_in_]))]
    
    # Convert the Scikit-learn model to ONNX
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    
    # Save the ONNX model to file
    model_file = os.path.join(MODELS_FOLDER, f'{model_name}.onnx')
    with open(model_file, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    
    return model_file


def load_onnx_model(model_name):
    """
    Load a model saved in ONNX format.
    
    Args:
    - model_name: Name of the model to load
    
    Returns:
    - onnx_session: ONNX Runtime inference session
    """
    model_file = os.path.join(MODELS_FOLDER, f'{model_name}.onnx')
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"No model found with the name '{model_name}' in ONNX format.")
    
    # Load the ONNX model into an inference session
    onnx_session = ort.InferenceSession(model_file)
    return onnx_session


def make_prediction_with_onnx(onnx_session, input_data):
    """
    Make predictions using an ONNX model.
    
    Args:
    - onnx_session: ONNX Runtime inference session
    - input_data: Input data to use for prediction
    
    Returns:
    - prediction: Predicted output from the ONNX model
    """
    input_name = onnx_session.get_inputs()[0].name
    input_data = np.array(input_data).astype(np.float32)
    prediction = onnx_session.run(None, {input_name: input_data})[0]
    return prediction


def convert_to_tflite(model, model_name):
    """
    Convert a TensorFlow model to TensorFlow Lite (TFLite) format.
    
    Args:
    - model: Trained Keras model to convert
    - model_name: Name to use for the saved TFLite model file
    
    Returns:
    - tflite_file: Path to the saved TFLite model file
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Save the TFLite model to file
    tflite_file = os.path.join(MODELS_FOLDER, f'{model_name}.tflite')
    with open(tflite_file, 'wb') as f:
        f.write(tflite_model)
    
    return tflite_file


def load_tflite_model(model_name):
    """
    Load a TensorFlow Lite model.
    
    Args:
    - model_name: Name of the TFLite model to load
    
    Returns:
    - interpreter: TFLite interpreter for the loaded model
    """
    tflite_file = os.path.join(MODELS_FOLDER, f'{model_name}.tflite')
    if not os.path.exists(tflite_file):
        raise FileNotFoundError(f"No model found with the name '{model_name}' in TensorFlow Lite format.")
    
    interpreter = tf.lite.Interpreter(model_path=tflite_file)
    interpreter.allocate_tensors()
    return interpreter


def make_prediction_with_tflite(interpreter, input_data):
    """
    Make predictions using a TensorFlow Lite model.
    
    Args:
    - interpreter: TFLite interpreter
    - input_data: Input data to use for prediction
    
    Returns:
    - prediction: Predicted output from the TFLite model
    """
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Prepare the input data
    input_data = np.array(input_data, dtype=np.float32)
    
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Invoke the interpreter
    interpreter.invoke()
    
    # Get the output prediction
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data


def list_saved_models():
    """
    List all saved models in the models directory.
    
    Returns:
    - model_files: List of all model files in the models directory
    """
    model_files = os.listdir(MODELS_FOLDER)
    return model_files
