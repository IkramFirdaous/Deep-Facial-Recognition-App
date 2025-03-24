from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger
from kivy.core.window import Window
from kivy.utils import get_color_from_hex

import tensorflow as tf
import os 
import cv2 
from layers import L1Dist
import numpy as np 

# Set app window background color (light pink)
Window.clearcolor = get_color_from_hex("#ffe6f0")  # light pastel pink

class CamApp(App):
    def build(self):
        # Main layout
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Webcam Display
        self.webcam = Image(size_hint=(1, 0.75))
        layout.add_widget(self.webcam)

        # Verification Label with girly font/color
        self.verification_label = Label(
            text="Hi girlie :) , identify yourself queen :3",
            size_hint=(1, 0.1),
            font_size='20sp',
            bold=True,
            color=get_color_from_hex("#cc3399")  # deep pink text
        )
        layout.add_widget(self.verification_label)

        # Pretty Button
        self.button = Button(
            text=" Verify Me ",
            size_hint=(1, 0.15),
            background_normal='',
            background_color=get_color_from_hex("#ffb3d9"),  # pastel pink
            color=get_color_from_hex("#660066"),  # dark purple text
            font_size='20sp',
            bold=True
        )
        self.button.bind(on_press=self.verify)
        layout.add_widget(self.button)

        # Load TensorFlow model
        self.model = tf.keras.models.load_model('siamesemodel.h5', custom_objects={'L1Dist': L1Dist})

        # Set up webcam
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 33.0)

        return layout

    def update(self, *args):
        ret, frame = self.capture.read()
        frame = frame[120:370, 200:450, :]

        buf = cv2.flip(frame, 0).tobytes()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.webcam.texture = img_texture

    def preprocess(self, file_path):
        byte_img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(byte_img)
        img = tf.image.resize(img, (100, 100))
        img = img / 255.0
        return img

    def verify(self, *args):
        detection_threshold = 0.5
        verification_threshold = 0.5

        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[120:370, 200:450, :]
        cv2.imwrite(SAVE_PATH, frame)

        results = []
        for image in os.listdir(os.path.join('application_data', 'verfication_images')):
            input_img = self.preprocess(SAVE_PATH)
            validation_img = self.preprocess(os.path.join('application_data', 'verfication_images', image))
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)

        detection = np.sum(np.array(results) > detection_threshold)
        verification = detection / len(os.listdir(os.path.join('application_data', 'verfication_images')))
        verified = verification > verification_threshold

        # Update label with cute message
        if verified:
            self.verification_label.text = " Verified! Welcome <3"
            self.verification_label.color = get_color_from_hex("#33cc99")  # mint green
        else:
            self.verification_label.text = " Unverified :(. Try Again!"
            self.verification_label.color = get_color_from_hex("#ff6666")  # soft red

        Logger.info(results)
        Logger.info(np.sum(np.array(results) > 0.2))
        Logger.info(np.sum(np.array(results) > 0.4))
        Logger.info(np.sum(np.array(results) > 0.6))
        Logger.info(np.sum(np.array(results) > 0.8))

        return results, verified

if __name__ == '__main__':
    CamApp().run()
