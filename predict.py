import cv2
import numpy as np
import tensorflow as tf

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='cloth_classifier_model.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (150, 150))  # Resize image to match model input
    image = image.astype(np.float32)       # Convert image to float32
    image /= 255.0                         # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Initialize the camera
camera = cv2.VideoCapture(1)  # Use default camera

while True:
    ret, frame = camera.read()
    if not ret:
        break

    # Preprocess the image
    input_data = preprocess_image(frame)

    # Set the tensor to the input data
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the results
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output_data[0])

    # Convert prediction to class label
    class_labels = ['handloom', 'normal_or_powerloom']  # Update based on your class labels
    predicted_class = class_labels[prediction]

    # Display the result
    cv2.putText(frame, f'Prediction: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Prediction Window', frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
camera.release()
cv2.destroyAllWindows()
