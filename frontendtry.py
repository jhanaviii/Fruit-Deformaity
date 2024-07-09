import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classifier")
        self.root.configure(bg="#FFB6C1")  # Light pink background

        # Load the trained model
        self.model = tf.keras.models.load_model(r'/Users/jhanaviagarwal/PycharmProjects/FruitDeformity/my_model.h5')

        # Create GUI elements
        self.canvas = tk.Canvas(root, width=300, height=300, bg='white')
        self.canvas.grid(row=0, column=0, columnspan=2, padx=20, pady=20)

        self.browse_button = tk.Button(root, text="Browse Image", command=self.browse_image, bg="#d0e7f9", fg="white", font=("Helvetica", 12, "bold"))
        self.browse_button.grid(row=1, column=0, padx=20, pady=20)

        self.classify_button = tk.Button(root, text="Classify", command=self.classify_image, bg="#d0e7f9", fg="white", font=("Helvetica", 12, "bold"))
        self.classify_button.grid(row=1, column=1, padx=20, pady=20)

        self.label = tk.Label(root, text="", bg="#FFB6C1", font=("Helvetica", 12))
        self.label.grid(row=2, column=0, columnspan=2, padx=20, pady=20)

    def browse_image(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if self.file_path:
            self.load_image()

    def load_image(self):
        try:
            self.img = Image.open(self.file_path)
            self.img = self.img.resize((300, 300))
            self.img = ImageTk.PhotoImage(self.img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def classify_image(self):
        if hasattr(self, 'file_path'):
            try:
                image = tf.keras.preprocessing.image.load_img(self.file_path, target_size=(128, 128))
                input_arr = tf.keras.preprocessing.image.img_to_array(image)
                input_arr = np.array([input_arr])  # Convert single image to a batch.
                predictions = self.model.predict(input_arr)
                class_names = ['Normal', 'Deformed']
                prediction = class_names[np.argmax(predictions)]
                quality_level = np.max(predictions)
                self.label.config(text=f"Prediction: {prediction}\nQuality Level: {quality_level:.2f}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to classify image: {e}")
        else:
            messagebox.showerror("Error", "Please browse an image first.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()
