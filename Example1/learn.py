# learn.py
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_and_save_model(model_save_path):
    # Load CIFAR-10 data
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    
    # Preprocess data
    train_images, test_images = train_images / 255.0, test_images / 255.0
    train_labels = train_labels.flatten()
    test_labels = test_labels.flatten()
    
    model = build_model()
    
    # Train the model
    model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_data=(test_images, test_labels))
    
    # Save the model
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    model_save_path = 'cifar10_model.h5'  # Path to save the trained model
    train_and_save_model(model_save_path)
