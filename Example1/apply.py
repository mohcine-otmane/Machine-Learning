# apply.py
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import os

class ImageClassifierApp:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = tf.keras.models.load_model(model_path)
        self.root = tk.Tk()
        self.root.title("Image Classifier")

        self.canvas = tk.Canvas(self.root, width=250, height=250)
        self.canvas.pack()

        self.label = tk.Label(self.root, text="")
        self.label.pack()

        self.green_button = tk.Button(self.root, text="Yes", command=self.correct_label)
        self.green_button.pack(side=tk.LEFT, padx=5)

        self.red_button = tk.Button(self.root, text="No", command=self.incorrect_label)
        self.red_button.pack(side=tk.LEFT, padx=5)

        self.load_image_button = tk.Button(self.root, text="Load Image", command=self.load_image)
        self.load_image_button.pack(side=tk.LEFT, padx=5)

        self.incorrect_data = []

    def preprocess_image(self, image_path):
        img = Image.open(image_path).resize((32, 32), Image.Resampling.LANCZOS)
        img_array = np.array(img) / 255.0
        return img_array[np.newaxis, ...]

    def display_image(self, img_path):
        img = Image.open(img_path).resize((250, 250), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.root.image = img_tk

    def load_image(self):
        img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if img_path:
            self.display_image(img_path)
            img_array = self.preprocess_image(img_path)
            predictions = self.model.predict(img_array)
            predicted_label = np.argmax(predictions[0])
            self.current_image_path = img_path
            self.current_predicted_label = predicted_label
            self.label.config(text=f"Predicted Label: {predicted_label}")

    def correct_label(self):
        messagebox.showinfo("Feedback", "Label confirmed. No action needed.")

    def incorrect_label(self):
        def submit_label():
            try:
                correct_label = int(entry.get())
                if correct_label < 0 or correct_label > 9:
                    raise ValueError
                self.incorrect_data.append((self.current_image_path, correct_label))
                self.update_model()
                messagebox.showinfo("Model Updated", "The model has been updated with the corrected label.")
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a valid label between 0 and 9.")
        
        self.incorrect_data.append((self.current_image_path, None))
        top = tk.Toplevel(self.root)
        top.title("Enter Correct Label")
        tk.Label(top, text="Enter Correct Label (0-9):").pack(pady=10)
        entry = tk.Entry(top)
        entry.pack(pady=10)
        tk.Button(top, text="Submit", command=submit_label).pack(pady=10)
        
    def update_model(self):
        (X_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
        X_train = X_train / 255.0
        y_train = y_train.flatten()

        # Append new incorrect data
        for img_path, class_index in self.incorrect_data:
            if class_index is not None:
                img_array = self.preprocess_image(img_path)
                X_train = np.append(X_train, img_array, axis=0)
                y_train = np.append(y_train, [class_index])

        y_train = y_train.flatten()
        
        model = self.build_model()
        model.fit(X_train, y_train, epochs=5, batch_size=64)
        model.save(self.model_path)
        print("Model updated and saved.")
        self.incorrect_data = []  # Clear incorrect data after updating

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    model_path = 'cifar10_model.h5'
    classifier = ImageClassifierApp(model_path)
    classifier.run()
