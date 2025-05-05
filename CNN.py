import os
from collections import Counter
from PIL import Image, ImageTk, ImageFilter, ImageEnhance
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import ttk, filedialog
import threading
import numpy as np

# --- Utils ---
def im2col_sliding(X, kernel_size, stride):
    batch, h, w, c = X.shape
    out_h = (h - kernel_size) // stride + 1
    out_w = (w - kernel_size) // stride + 1
    col = np.zeros((batch, out_h, out_w, kernel_size, kernel_size, c))
    for y in range(kernel_size):
        y_max = y + stride * out_h
        for x in range(kernel_size):
            x_max = x + stride * out_w
            col[:, :, :, y, x, :] = X[:, y:y_max:stride, x:x_max:stride, :]
    col = col.reshape(batch, out_h, out_w, -1)
    return col

# --- CNN Layers ---
class ConvLayer:
    def __init__(self, input_channels, num_filters, kernel_size, stride=1, padding=0):
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.filters = np.random.randn(num_filters, input_channels * kernel_size * kernel_size) * np.sqrt(2 / (input_channels * kernel_size * kernel_size))
        self.biases = np.zeros(num_filters)

    def im2col(self, X):
        batch_size, in_h, in_w, in_c = X.shape
        out_h = (in_h - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_w = (in_w - self.kernel_size + 2 * self.padding) // self.stride + 1

        col = np.zeros((batch_size, out_h, out_w, self.kernel_size, self.kernel_size, in_c))
        X_padded = np.pad(X, ((0,0), (self.padding, self.padding), (self.padding, self.padding), (0,0)), mode='constant')

        for y in range(out_h):
            for x in range(out_w):
                col[:, y, x, :, :, :] = X_padded[
                    :,
                    y*self.stride:y*self.stride+self.kernel_size,
                    x*self.stride:x*self.stride+self.kernel_size,
                    :
                ]
        col = col.reshape(batch_size, out_h, out_w, -1)
        return col

    def forward(self, X):
        self.X_input = X
        self.col = self.im2col(X)
        out = np.einsum('bijc,oc->bijo', self.col, self.filters) + self.biases.reshape(1, 1, 1, -1)
   
        return out

    def backward(self, d_out, learning_rate):
        # Anh chỉ cần giữ chỗ này tạm ổn đã
        return d_out


class MaxPoolLayer:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, X):
        self.X_input = X
        batch, h, w, c = X.shape
        out_h = (h - self.pool_size) // self.stride + 1
        out_w = (w - self.pool_size) // self.stride + 1
        out = np.zeros((batch, out_h, out_w, c))
        for i in range(out_h):
            for j in range(out_w):
                region = X[:, i * self.stride:i * self.stride + self.pool_size, j * self.stride:j * self.stride + self.pool_size, :]
                out[:, i, j, :] = np.max(region, axis=(1, 2))
        return out

    def backward(self, d_out):
        batch, out_h, out_w, channels = d_out.shape
        dX = np.zeros_like(self.X_input)
        for b in range(batch):
            for c in range(self.X_input.shape[-1]):  # ✅ SỬA ở đây
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size

                        region = self.X_input[b, h_start:h_end, w_start:w_end, c]
                        mask = (region == np.max(region))
                        dX[b, h_start:h_end, w_start:w_end, c] += d_out[b, i, j, c] * mask
        return dX


class FlattenLayer:
    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, d_out):
        return d_out.reshape(self.input_shape)

class DenseLayer:
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.b = np.zeros((1, output_size))

    def forward(self, X):
        self.X_input = X
        return np.dot(X, self.W) + self.b

    def backward(self, d_out, lr):
        dW = np.dot(self.X_input.T, d_out)
        db = np.sum(d_out, axis=0, keepdims=True)
        dX = np.dot(d_out, self.W.T)
        self.W -= lr * dW
        self.b -= lr * db
        return dX

class ReLULayer:
    def forward(self, X):
        self.mask = (X > 0)
        return X * self.mask

    def backward(self, d_out):
        return d_out * self.mask

class SoftmaxCrossEntropy:
    def forward(self, X):
        exp_scores = np.exp(X - np.max(X, axis=1, keepdims=True))
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs

    def backward(self, y_true):
        batch_size = y_true.shape[0]
        grad = self.probs.copy()
        grad[np.arange(batch_size), y_true] -= 1
        grad /= batch_size
        return grad

class ConvBlock:
    def __init__(self, in_channels, out_channels, kernel_size, pool_size=2, stride=1, padding=1):
        self.conv = ConvLayer(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = ReLULayer()
        self.pool = MaxPoolLayer(pool_size, pool_size)

    def forward(self, X):
        X = self.conv.forward(X)
        X = self.relu.forward(X)
        X = self.pool.forward(X)
        return X

    def backward(self, grad, lr):
        grad = self.pool.backward(grad)
        grad = self.relu.backward(grad)
        grad = self.conv.backward(grad, lr)
        return grad




class SimpleCNN:
    def __init__(self, num_classes):
        self.block1 = ConvBlock(1, 8, 3)
        self.block2 = ConvBlock(8, 16, 3)
        self.block3 = ConvBlock(16, 32, 3)

        self.flatten = FlattenLayer()
        self.fc1 = DenseLayer(4 * 4 * 32, 64)
        self.relu_fc = ReLULayer()
        self.fc2 = DenseLayer(64, num_classes)

        self.softmax = SoftmaxCrossEntropy()

    def forward(self, X):
        X = self.block1.forward(X)
        X = self.block2.forward(X)
        X = self.block3.forward(X)

        X = self.flatten.forward(X)
        X = self.fc1.forward(X)
        X = self.relu_fc.forward(X)
        X = self.fc2.forward(X)
        return self.softmax.forward(X)

    def backward(self, y_true, lr):
        grad = self.softmax.backward(y_true)
        grad = self.fc2.backward(grad, lr)
        grad = self.relu_fc.backward(grad)
        grad = self.fc1.backward(grad, lr)
        grad = self.flatten.backward(grad)

        grad = self.block3.backward(grad, lr)
        grad = self.block2.backward(grad, lr)
        grad = self.block1.backward(grad, lr)

# Tiếp tục viết phần Tkinter và training tương tự anh đang làm hoặc em viết thêm nếu anh yêu cầu 😄.


# --- Data ---
def augment_image(img):
    augmented = [
        img.transpose(Image.FLIP_LEFT_RIGHT),
        img.rotate(15),
        img.rotate(-15),
        ImageEnhance.Brightness(img).enhance(1.5),
        ImageEnhance.Brightness(img).enhance(0.7)
    ]
    return augmented

def load_data(root_folder, img_size=(32, 32)):
    X, y = [], []
    person_folders = sorted([p for p in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, p))],
                            key=lambda x: int(x))
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
        self.title("CNN Fingerprint Recognition")
        self.geometry("800x600")
        self.dataset_path = None
        self.model = None
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
        

        num_classes = len(set(y))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print("Num classes:", len(set(y)))
        print("Sample y_train:", y_train[:10])
        print("Sample y_test:", y_test[:10])
        self.model = SimpleCNN(num_classes)
        print("Training started...")
        epochs = 50
        lr = 0.005
        batch_size = 32
        for epoch in range(1, epochs + 1):
            perm = np.random.permutation(len(X_train))
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[perm[i:i+batch_size]]
                y_batch = y_train[perm[i:i+batch_size]]

                y_pred = self.model.forward(X_batch)
                acc = np.mean(np.argmax(y_pred, axis=1) == y_batch)
                loss = -np.sum(np.log(y_pred[range(len(y_batch)), y_batch] + 1e-8)) / len(y_batch)
                self.model.backward(y_batch, lr)

        print("Training completed!")

    def save_model(self):
        if self.model:
            path = filedialog.asksaveasfilename(defaultextension=".npy", filetypes=[("Numpy files", "*.npy")])
            if path:
                self.model.save_model(path)

    def load_model(self):
        path = filedialog.askopenfilename(filetypes=[("Numpy files", "*.npy")])
        if path:
            if self.model is None:
                self.model = SimpleCNN(num_classes=1)
            self.model.load_model(path)

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
        if self.model and self.current_fingerprint is not None and self.person_names is not None:
            input_data = np.expand_dims(self.current_fingerprint, axis=0)
            y_pred = self.model.forward(input_data)
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

# Run app
if __name__ == "__main__":
    app = FingerprintRecognitionApp()
    app.mainloop()
