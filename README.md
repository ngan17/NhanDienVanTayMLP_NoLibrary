🌀 Fingerprint Recognition using MLP + Tkinter GUI
This project is a Fingerprint Recognition System built with:

🧠 MLP (Multi-Layer Perceptron) for classification.

🖼️ Tkinter GUI for an interactive user interface.

🛠️ Augmented data with rotation to improve robustness.

🚀 Features
✅ Train a fingerprint dataset with:

Auto data augmentation (rotate each image from 10° to 180°).

Dropout to prevent overfitting.

Batch training with progress bar and Loss/Accuracy chart.

✅ Recognize fingerprints:

Select any .BMP image.

View the prediction and the Top 3 classes with highest confidence.

Real-time image processing: blur & rotate the input fingerprint.
💻 Tech stack
Module	Usage
Tkinter	GUI: buttons, progress bar, display
NumPy	Vector/matrix computation
PIL (Pillow)	Image loading, resizing, augmentation
Scikit-learn	Train/test split
Matplotlib	Training chart (Loss & Accuracy)
Threading	Run training in background

🔧 How it works
1️⃣ Load Dataset:

Reads all .BMP files inside each person's folder.

Resizes images to 64×64 grayscale.

Augments data by rotating each image (from 10° to 180°).

2️⃣ MLP Model:

Input size: 4096 (64×64).

Hidden layer: 256 units with ReLU + dropout.

Output: softmax with as many classes as there are persons.

3️⃣ Train:

Batch size: 32

Learning rate: 0.01

Default: 100 epochs

Shows real-time progress and chart of training performance.

4️⃣ Recognition:

After training, choose any fingerprint to predict.

The app shows the top 3 predicted persons + their confidence scores.

📸 Screenshots
Train model	Predict fingerprint

(You can replace these with real screenshots later.)

▶️ How to run
1️⃣ Install dependencies:

bash
Copy
Edit
pip install numpy pillow scikit-learn matplotlib
2️⃣ Prepare your dataset in the correct folder structure.

3️⃣ Run the app:

bash
Copy
Edit
python main.py
💡 Extensions (Ideas to improve)
Add more image augmentations: flip, brightness, noise.

Support real-time camera fingerprint capture.

Save & load trained model.

Switch from MLP to CNN for better accuracy.

✨ Credit
Created by [nganit17] – AI Fingerprint Recognition Project 🚀.
