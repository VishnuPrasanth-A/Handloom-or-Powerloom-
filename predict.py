import cv2
import numpy as np
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path='cloth_classifier_model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def preprocess_image(image):
    image = cv2.resize(image, (150, 150))  
    image = image.astype(np.float32)       
    image /= 255.0                         
    image = np.expand_dims(image, axis=0)  
    return image


camera = cv2.VideoCapture(1)  

while True:
    ret, frame = camera.read()
    if not ret:
        break

   
    input_data = preprocess_image(frame)

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output_data[0])

    class_labels = ['handloom', 'normal_or_powerloom']  # Update based on your class labels
    predicted_class = class_labels[prediction]

    cv2.putText(frame, f'Prediction: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Prediction Window', frame)

  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
