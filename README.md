# Facial-Recognition-App
 
A real-time face verification application using Kivy for the graphical interface, OpenCV for webcam integration, and a Siamese neural network built with TensorFlow/Keras for identity verification.

Features
Face verification via a pre-trained Siamese neural network

Live webcam feed using OpenCV

User-friendly graphical interface with Kivy

Real-time image capture and verification

Project Structure
bash
Copier
Modifier
├── application_data/

│   ├── input_image/

│   │   └── input_image.jpg  # Captured from webcam

│   └── verification_images/      # Stored reference images

├── layers.py                     # Custom L1 distance layer

├── siamesemodel.h5               # Pretrained Siamese model

├── main.py                       # Main Kivy app script

├── Face_Recong_notebook.ipynb    # Model training notebook

└── README.md
Installation
Clone the repository:

bash
Copier
Modifier
git clone https://github.com/yourusername/face-verification-app.git
cd face-verification-app
Install required packages:

bash
Copier
Modifier
pip install kivy tensorflow opencv-python numpy
Run the application:

bash
Copier
Modifier
python main.py
How It Works
The application captures an image from your webcam.

It compares the captured image with stored reference images using a Siamese neural network.

The similarity between images is calculated using a custom L1 distance layer.

If the similarity exceeds the set threshold, the user is marked as Verified, otherwise Unverified.

User Interface Design
Light pastel color scheme for a clean and soft look

Simplified layout using vertical box design

Button and label customization for improved visual appeal

Future Enhancements
Support for multiple user profiles

Enhanced UI with KivyMD or custom widgets

Secure storage and management of verification data

Packaging as a standalone desktop application

