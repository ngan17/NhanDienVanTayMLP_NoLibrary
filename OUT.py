import os
import numpy as np
from PIL import Image, ImageTk, ImageFilter
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
import threading

# 1. Load data
def load_data(root_folder, img_size=(64, 64)):
    X = []
    y = []
    person_folders = os.listdir(root_folder)
    person_folders = [p for p in person_folders if os.path.isdir(os.path.join(root_folder, p))]
    person_folders.sort()
    person_names = person_folders  # Lưu tên của từng người
    label_mapping = {name: idx for idx, name in enumerate(person_folders)}
    
    for person_id in person_folders:
        person_path = os.path.join(root_folder, person_id, "Fingerprint")
        for img_name in os.listdir(person_path):
            if img_name.endswith('.BMP'):
                img_path = os.path.join(person_path, img_name)
                img = Image.open(img_path).convert('L')
                img = img.resize(img_size)
                
                # 1. Ảnh gốc
                img_array = np.array(img).flatten() / 255.0
                X.append(img_array)
                y.append(label_mapping[person_id])

                # 2. Augmentation: Xoay từ 1° đến 180° mỗi 10°
                for angle in range(10, 181, 10):
                    rotated_img = img.rotate(angle)
                    rotated_array = np.array(rotated_img).flatten() / 255.0
                    X.append(rotated_array)
                    y.append(label_mapping[person_id])

    return np.array(X), np.array(y), np.array(person_names)


# 2. Định nghĩa MLP (thêm dropout)
class SimpleMLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X, training=True, dropout_rate=0.2):
        self.z1 = X.dot(self.W1) + self.b1
        self.a1 = self.relu(self.z1)

        if training and dropout_rate > 0:
            self.dropout_mask = (np.random.rand(*self.a1.shape) > dropout_rate).astype(float)
            self.a1 *= self.dropout_mask  # Apply dropout

        self.z2 = self.a1.dot(self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(m), y_true] + 1e-9)  # avoid log(0)
        loss = np.sum(log_likelihood) / m
        return loss

    def backward(self, X, y_true, y_pred, learning_rate=0.01):
        m = y_true.shape[0]
        dz2 = y_pred.copy()
        dz2[range(m), y_true] -= 1
        dz2 /= m

        dW2 = self.a1.T.dot(dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2.dot(self.W2.T)
        dz1 = da1 * self.relu_derivative(self.z1)

        # Apply dropout mask in backward if used
        if hasattr(self, 'dropout_mask'):
            dz1 *= self.dropout_mask

        dW1 = X.T.dot(dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2


# 3. Main App
class FingerprintRecognitionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Fingerprint Recognition (Optimized)")
        self.geometry("850x650")

        self.dataset_path = None
        self.model = None
        self.current_fingerprint = None
        self.person_names = None

        self.loss_history = []
        self.acc_history = []

        self.create_widgets()

    def create_widgets(self):
        dataset_frame = ttk.Frame(self)
        dataset_frame.pack(pady=20)

        dataset_label = ttk.Label(dataset_frame, text="Dataset Path:")
        dataset_label.pack(side=tk.LEFT)

        self.dataset_entry = ttk.Entry(dataset_frame, width=50)
        self.dataset_entry.pack(side=tk.LEFT)

        dataset_button = ttk.Button(dataset_frame, text="Browse", command=self.browse_dataset)
        dataset_button.pack(side=tk.LEFT, padx=10)

        train_frame = ttk.Frame(self)
        train_frame.pack(pady=20)

        train_button = ttk.Button(train_frame, text="Train Model", command=self.train_model_thread)
        train_button.pack(side=tk.LEFT)

        view_chart_button = ttk.Button(train_frame, text="View Training Chart", command=self.view_training_chart)
        view_chart_button.pack(side=tk.LEFT, padx=10)

        self.progress = ttk.Progressbar(train_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progress.pack(side=tk.LEFT, padx=10)

        recognize_frame = ttk.Frame(self)
        recognize_frame.pack(pady=20)

        self.image_label = ttk.Label(recognize_frame)
        self.image_label.pack(side=tk.LEFT, padx=10)

        recognize_button = ttk.Button(recognize_frame, text="Recognize", command=self.recognize_fingerprint)
        recognize_button.pack(side=tk.LEFT, padx=10)

        self.result_label = ttk.Label(recognize_frame, text="")
        self.result_label.pack(side=tk.LEFT)

        fingerprint_frame = ttk.Frame(self)
        fingerprint_frame.pack(pady=20)

        fingerprint_label = ttk.Label(fingerprint_frame, text="Fingerprint Image:")
        fingerprint_label.pack(side=tk.LEFT)

        self.fingerprint_button = ttk.Button(fingerprint_frame, text="Choose Image", command=self.choose_fingerprint)
        self.fingerprint_button.pack(side=tk.LEFT, padx=10)

        image_processing_frame = ttk.Frame(self)
        image_processing_frame.pack(pady=20)

        self.blur_scale = ttk.Scale(image_processing_frame, from_=0, to=10, orient=tk.HORIZONTAL, command=self.apply_blur)
        self.blur_scale.pack(side=tk.LEFT, padx=10)

        self.rotate_scale = ttk.Scale(image_processing_frame, from_=-180, to=180, orient=tk.HORIZONTAL, command=self.apply_rotate)
        self.rotate_scale.pack(side=tk.LEFT, padx=10)

    def browse_dataset(self):
        self.dataset_path = filedialog.askdirectory()
        self.dataset_entry.delete(0, tk.END)
        self.dataset_entry.insert(0, self.dataset_path)

    def train_model_thread(self):
        threading.Thread(target=self.train_model).start()

    def train_epoch(self, X_train, y_train, batch_size=32, learning_rate=0.01):
        idx = np.arange(X_train.shape[0])
        np.random.shuffle(idx)
        X_train = X_train[idx]
        y_train = y_train[idx]

        for start in range(0, X_train.shape[0], batch_size):
            end = start + batch_size
            X_batch = X_train[start:end]
            y_batch = y_train[start:end]

            y_pred = self.model.forward(X_batch, training=True)
            loss = self.model.compute_loss(y_batch, y_pred)
            self.model.backward(X_batch, y_batch, y_pred, learning_rate)

    def train_model(self):
        if self.dataset_path:
            X, y, self.person_names = load_data(self.dataset_path)
            num_classes = len(set(y))
            input_size = X.shape[1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            self.model = SimpleMLP(input_size=input_size, hidden_size=256, output_size=num_classes)

            self.loss_history = []
            self.acc_history = []

            total_epochs = 100
            self.progress['maximum'] = total_epochs

            for epoch in range(total_epochs):
                self.train_epoch(X_train, y_train, batch_size=32, learning_rate=0.01)

                # Evaluate on training set
                y_pred = self.model.forward(X_train, training=False)
                loss = self.model.compute_loss(y_train, y_pred)
                acc = np.mean(np.argmax(y_pred, axis=1) == y_train)
                self.loss_history.append(loss)
                self.acc_history.append(acc)

                if epoch % 10 == 0:
                    print(f"Epoch {epoch} - Loss: {loss:.4f} - Accuracy: {acc:.4f}")

                self.progress['value'] = epoch + 1
                self.update_idletasks()

            print("Model trained successfully!")
        else:
            print("Please select a dataset path first.")

    def view_training_chart(self):
        if self.loss_history:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.plot(self.loss_history, label="Loss")
            plt.title("Loss over Epochs")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(self.acc_history, label="Accuracy", color='green')
            plt.title("Accuracy over Epochs")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()

            plt.tight_layout()
            plt.show()
        else:
            print("No training history available yet.")

    def choose_fingerprint(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.BMP")])
        if file_path:
            img = Image.open(file_path).convert('L')
            img = img.resize((64, 64))

            # img = self.apply_image_processing(img)

            self.current_fingerprint = np.array(img).flatten() / 255.0

            fingerprint_photo = ImageTk.PhotoImage(img)
            self.image_label.configure(image=fingerprint_photo)
            self.image_label.image = fingerprint_photo

            self.recognize_fingerprint()

    def apply_image_processing(self, img):
        blur_value = self.blur_scale.get()

        rotate_value = self.rotate_scale.get()
        rotate_value = round(rotate_value / 10) * 10  # làm tròn bội số 10

        img = img.filter(ImageFilter.GaussianBlur(blur_value))
        img = img.rotate(rotate_value)

        return img


    def apply_blur(self, value):
        if self.current_fingerprint is not None:
            img = self.apply_image_processing(Image.fromarray((self.current_fingerprint.reshape(64, 64) * 255).astype(np.uint8)))
            fingerprint_photo = ImageTk.PhotoImage(img)
            self.image_label.configure(image=fingerprint_photo)
            self.image_label.image = fingerprint_photo

    def apply_rotate(self, value):
        if self.current_fingerprint is not None:
            img = self.apply_image_processing(Image.fromarray((self.current_fingerprint.reshape(64, 64) * 255).astype(np.uint8)))
            fingerprint_photo = ImageTk.PhotoImage(img)
            self.image_label.configure(image=fingerprint_photo)
            self.image_label.image = fingerprint_photo

    def recognize_fingerprint(self):
        if self.model and self.current_fingerprint is not None and self.person_names is not None:
            fingerprint_array = self.current_fingerprint.reshape(1, -1)
            y_pred = self.model.forward(fingerprint_array, training=False)
            
            predicted_class = np.argmax(y_pred)
            predicted_name = self.person_names[predicted_class]
            predicted_prob = y_pred[0, predicted_class]

            # Lấy top 3 xác suất cao nhất
            top_indices = np.argsort(y_pred[0])[::-1][:3]

            result_text = f"Predicted: {predicted_name} ({predicted_prob * 100:.2f}%)\n\n"
            result_text += "Top 3 probabilities:\n"

            for idx in top_indices:
                name = self.person_names[idx]
                prob = y_pred[0, idx]
                result_text += f"{name}: {prob * 100:.2f}%\n"

            self.result_label.configure(text=result_text)
        else:
            self.result_label.configure(text="No fingerprint image available.")




if __name__ == "__main__":
    app = FingerprintRecognitionApp()
    app.mainloop()
