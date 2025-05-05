import os
from collections import Counter
from PIL import Image, ImageTk, ImageFilter, ImageEnhance
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import ttk, filedialog
import threading
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# --- Build Keras CNN ---

def build_keras_cnn(input_shape=(32, 32, 1), num_classes=45):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# --- Data Augmentation ---

def augment_image(img):
    augmented = []
    # Flip
    augmented.append(img.transpose(Image.FLIP_LEFT_RIGHT))
    # Rotate nhiều góc
    for angle in [5, 10, 15, 20, -5, -10, -15, -20]:
        augmented.append(img.rotate(angle))
    # Brightness
    enhancer = ImageEnhance.Brightness(img)
    augmented.append(enhancer.enhance(1.5))
    augmented.append(enhancer.enhance(0.7))
    # Contrast
    contrast = ImageEnhance.Contrast(img)
    augmented.append(contrast.enhance(1.5))
    augmented.append(contrast.enhance(0.7))
    # Blur (Gaussian)
    augmented.append(img.filter(ImageFilter.GaussianBlur(radius=1)))
    augmented.append(img.filter(ImageFilter.GaussianBlur(radius=2)))
    return augmented


# --- Load data function ---

def load_data(root_folder, img_size=(32, 32)):
    X, y = [], []
    person_folders = [p for p in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, p))]
    person_folders = sorted(person_folders, key=lambda x: int(x))
    label_mapping = {name: idx for idx, name in enumerate(person_folders)}

    for person_id in person_folders:
        person_path = os.path.join(root_folder, person_id, "Fingerprint")
        for img_name in os.listdir(person_path):
            if img_name.endswith('.BMP'):
                img_path = os.path.join(person_path, img_name)
                img = Image.open(img_path).convert('L').resize(img_size)
                img_array = np.array(img).reshape(img_size[0], img_size[1], 1) / 255.0
                X.append(img_array)
                y.append(label_mapping[person_id])

                # ➔ augment thêm:
                for aug_img in augment_image(img):
                    aug_img_resized = aug_img.resize(img_size)
                    aug_array = np.array(aug_img_resized).reshape(img_size[0], img_size[1], 1) / 255.0
                    X.append(aug_array)
                    y.append(label_mapping[person_id])
    print("Counter of y:", Counter(y))
    return np.array(X), np.array(y), np.array(person_folders)

# --- Tkinter App ---

class FingerprintRecognitionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CNN Fingerprint Recognition (Keras)")
        self.geometry("800x600")
        self.dataset_path = None
        self.keras_model = None
        self.person_names = None
        self.current_fingerprint = None

        self.create_widgets()

    def create_widgets(self):
        dataset_frame = ttk.Frame(self)
        dataset_frame.pack(pady=10)
        ttk.Label(dataset_frame, text="Dataset Path:").pack(side=tk.LEFT)
        self.dataset_entry = ttk.Entry(dataset_frame, width=50)
        self.dataset_entry.pack(side=tk.LEFT)
        ttk.Button(dataset_frame, text="Browse", command=self.browse_dataset).pack(side=tk.LEFT, padx=5)

        train_frame = ttk.Frame(self)
        train_frame.pack(pady=10)
        ttk.Button(train_frame, text="Train Model", command=self.train_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(train_frame, text="Save Model", command=self.save_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(train_frame, text="Load Model", command=self.load_model).pack(side=tk.LEFT, padx=5)

        recognize_frame = ttk.Frame(self)
        recognize_frame.pack(pady=10)
        self.image_label = ttk.Label(recognize_frame)
        self.image_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(recognize_frame, text="Choose Image", command=self.choose_fingerprint).pack(side=tk.LEFT, padx=5)
        self.result_label = ttk.Label(recognize_frame, text="")
        self.result_label.pack(side=tk.LEFT, padx=5)

        image_processing_frame = ttk.Frame(self)
        image_processing_frame.pack(pady=10)
        self.blur_scale = ttk.Scale(image_processing_frame, from_=0, to=10, orient=tk.HORIZONTAL, command=self.update_image_processing)
        self.blur_scale.pack(side=tk.LEFT, padx=10)
        self.rotate_scale = ttk.Scale(image_processing_frame, from_=-180, to=180, orient=tk.HORIZONTAL, command=self.update_image_processing)
        self.rotate_scale.pack(side=tk.LEFT, padx=10)

    def browse_dataset(self):
        self.dataset_path = filedialog.askdirectory()
        self.dataset_entry.delete(0, tk.END)
        self.dataset_entry.insert(0, self.dataset_path)

    def train_model(self):
        if not self.dataset_path:
            print("Please select a dataset path first.")
            return
        threading.Thread(target=self._train_model_thread, daemon=True).start()

    def _train_model_thread(self):
        print("Loading data...")
        X, y, self.person_names = load_data(self.dataset_path)
        print("Num classes:", len(set(y)))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # ➔ Build Keras model
        input_shape = X_train.shape[1:]
        num_classes = len(set(y))
        model = build_keras_cnn(input_shape, num_classes)

        print("Training started...")
        model.fit(X_train, y_train, epochs=100, batch_size=32,
                  validation_data=(X_test, y_test))

        self.keras_model = model
        print("Training completed!")

    def save_model(self):
        if self.keras_model:
            path = filedialog.asksaveasfilename(defaultextension=".h5", filetypes=[("H5 files", "*.h5")])
            if path:
                self.keras_model.save(path)
                print(f"Model saved to {path}")

    def load_model(self):
        path = filedialog.askopenfilename(filetypes=[("H5 files", "*.h5")])
        if path:
            self.keras_model = models.load_model(path)
            print(f"Model loaded from {path}")

    def choose_fingerprint(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.BMP")])
        if file_path:
            img = Image.open(file_path).convert('L').resize((32, 32))
            img = self.apply_image_processing(img)
            img_array = np.array(img).reshape(32, 32, 1) / 255.0
            self.current_fingerprint = img_array
            img_photo = ImageTk.PhotoImage(img)
            self.image_label.configure(image=img_photo)
            self.image_label.image = img_photo
            self.recognize_fingerprint()

    def recognize_fingerprint(self):
        if self.keras_model and self.current_fingerprint is not None and self.person_names is not None:
            input_data = np.expand_dims(self.current_fingerprint, axis=0)
            y_pred = self.keras_model.predict(input_data)
            predicted_class = np.argmax(y_pred)
            predicted_name = self.person_names[predicted_class]
            self.result_label.config(text=f"Predicted: {predicted_name}")

    def apply_image_processing(self, img):
        blur_value = self.blur_scale.get()
        if blur_value > 0:
            img = img.filter(ImageFilter.GaussianBlur(blur_value))
        rotate_value = self.rotate_scale.get()
        img = img.rotate(rotate_value)
        return img

    def update_image_processing(self, value):
        if self.current_fingerprint is not None:
            img = Image.fromarray((self.current_fingerprint.reshape(32, 32) * 255).astype(np.uint8))
            img = self.apply_image_processing(img)
            img_photo = ImageTk.PhotoImage(img)
            self.image_label.configure(image=img_photo)
            self.image_label.image = img_photo

# --- Run app ---

if __name__ == "__main__":
    app = FingerprintRecognitionApp()
    app.mainloop()
