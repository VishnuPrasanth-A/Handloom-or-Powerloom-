# Cloth Classification with TensorFlow Lite

## Overview

This project demonstrates how to classify cloth types into 'handloom' or 'normal_or_powerloom' using a Convolutional Neural Network (CNN) model trained in TensorFlow and converted to TensorFlow Lite for efficient inference. The application captures images using a webcam, processes them, and predicts the cloth type in real-time.

## Project Structure

- `model_training.py`: Script to train and save the TensorFlow model.
- `predict.py`: Script to run inference using the TensorFlow Lite model.
- `cloth_classifier_model.h5`: Trained Keras model (saved model file).
- `cloth_classifier_model.tflite`: TensorFlow Lite model (converted model file).

## Prerequisites

1. **Python**: Make sure you have Python 3.x installed.
2. **Required Libraries**: Install the required Python libraries using the following command:

    ```bash
    pip install tensorflow tflite-runtime opencv-python
    ```

3. **Dataset**: Prepare your dataset and organize it into the following directory structure:
    ```
    dataset/
    ├── train/
    │   ├── handloom/
    │   └── normal_or_powerloom/
    ├── validation/
    │   ├── handloom/
    │   └── normal_or_powerloom/
    └── test/
        ├── handloom/
        └── normal_or_powerloom/
    ```

## Model Training

### 1. Training the Model

Run the `model_training.py` script to train the CNN model and save it. This script sets up data augmentation, trains the model, evaluates it, and saves the final model.

```bash
python model_training.py
```

### 2. Converting the Model to TensorFlow Lite

If not already done, convert the Keras model to TensorFlow Lite format:

```python
import tensorflow as tf

# Load the trained Keras model
model = tf.keras.models.load_model('cloth_classifier_model.h5')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open('cloth_classifier_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

## Running Inference

### 1. Setting Up

Ensure you have a working webcam and the TensorFlow Lite model (`cloth_classifier_model.tflite`) is in the same directory as the `predict.py` script.

### 2. Running the Prediction Script

Run the `predict.py` script to start the real-time classification:

```bash
python predict.py
```

The script will open a window showing the live feed from the webcam with predictions displayed. Press 'q' to exit the script.

## Code Explanation

### Model Training (`model_training.py`)

- **Data Preparation**: Uses `ImageDataGenerator` to augment training data and preprocess images.
- **Model Architecture**: Defines a CNN with three convolutional layers followed by dense layers.
- **Training**: Compiles and fits the model with early stopping and checkpoint callbacks.
- **Evaluation**: Evaluates the model on the test set and plots training/validation accuracy and loss.

### Inference (`predict.py`)

- **Model Loading**: Loads the TensorFlow Lite model and sets up the interpreter.
- **Image Preprocessing**: Resizes, normalizes, and prepares images for model input.
- **Inference**: Runs the model on the captured image and displays the predicted class.
- **Webcam Integration**: Captures real-time video feed and overlays predictions.

## Troubleshooting

- **Camera Issues**: Ensure the webcam is properly connected and accessible.
- **Model Errors**: Verify that the TensorFlow Lite model path is correct.
- **Library Errors**: Check that all required libraries are installed and updated.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- TensorFlow: [tensorflow.org](https://www.tensorflow.org/)
- OpenCV: [opencv.org](https://opencv.org/)

## Contact

For any questions or issues, please contact [your-email@example.com](mailto:your-vishnuprasanth.a.agri44@gmail.com).

---

Feel free to customize the README file according to your specific project details and any additional instructions you may have.
