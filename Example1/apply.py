# apply.py
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

def load_model(model_path):
    # Load the saved model
    model = tf.keras.models.load_model(model_path)
    return model

def preprocess_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(32, 32))
    img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_image(model, img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions[0])
    return class_index

def display_image(img_path):
    img = plt.imread(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    model_path = 'cifar10_model.h5'  # Path to the saved model
    img_path = 'path_to_your_image.jpg'  # Path to the image you want to classify

    model = load_model(model_path)
    class_index = predict_image(model, img_path)

    print(f'Predicted class index: {class_index}')
    display_image(img_path)
