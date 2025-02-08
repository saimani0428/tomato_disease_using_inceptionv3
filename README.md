# tomato_leaf_disease_using_inceptionv3

📌 Overview

This project uses Inception V3 to detect diseases in tomato leaves. The model classifies images into different disease categories based on a deep learning approach.

📂 Dataset

The dataset consists of images of tomato leaves categorized into healthy and diseased conditions. You can download the dataset from:
🔗 https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf

⚙ Features

Deep Learning Model: Inception V3 (Transfer Learning)

Preprocessing: Image resizing, augmentation, normalization

Frameworks Used: TensorFlow, Keras, OpenCV

Evaluation Metrics: Accuracy, Precision, Recall, F1-score


🛠 Installation

install dependencies:

pip install -r requirements.txt

🔧 Preprocessing Steps

1. Convert images to RGB (if grayscale).


2. Resize images to  299x299 for Inception V3.


3. Normalize pixel values (0 to 1).


4. Apply data augmentation (rotation, flipping, brightness).



🚀 Model Training

Run the model on jupyter

🖼 Testing the Model

Run inference :

python app.py 

predicting :

upload the image in webpage

Example Output:

Predicted Class: Tomato Leaf Blight
Confidence: 92.5%

📊 Results

📝 To-Do

Improve accuracy with data augmentation.

Try alternative models like MobileNet or EfficientNet.


🤝 Contributing

Feel free to open an issue or submit a pull request!

