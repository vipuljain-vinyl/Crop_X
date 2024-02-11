# main.py
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.graphics.texture import Texture
from PIL import Image
import numpy as np
import onnxruntime as rt

class CropDiseaseApp(App):
    def build(self):
        self.load_model()
        layout = BoxLayout(orientation='vertical')
        self.camera = Camera(resolution=(640, 480), play=True)
        btn_capture = Button(text="Capture", size_hint=(1, 0.1))
        btn_capture.bind(on_press=self.capture)
        self.label = Label(text="Prediction: ")

        layout.add_widget(self.camera)
        layout.add_widget(btn_capture)
        layout.add_widget(self.label)
        return layout

    def load_model(self):
        # Load your ONNX model using onnxruntime
        self.model = rt.InferenceSession('disease_best_model.onnx')

    def capture(self, instance):
        texture = self.camera.texture
        texture.save('captured_image.png')

        # Preprocess the captured image and run inference
        image = Image.open('captured_image.png').resize((224, 224))
        input_data = np.array(image).transpose((2, 0, 1)).reshape(1, 3, 224, 224).astype(np.float32)
        result = self.model.run(None, {'input': input_data})

        # Process the inference result (replace this with your logic)
        prediction = result[0][0]

        # Update the label with the prediction
        self.label.text = f"Prediction: {prediction}"

if __name__ == '__main__':
    CropDiseaseApp().run()
