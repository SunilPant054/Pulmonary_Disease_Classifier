import gradio as gr
import numpy as np
import cv2


import gradio as gr

class Frontend:
    def __init__(self):
        self.model = None

    def set_model(self, model):
        self.model = model

    def preprocess_image(self, image):
        """
        Resize and normalize the uploaded image.
        """
        # We are using the same preprocess logic as in the Predict class
        image = cv2.resize(image, (256, 256))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        return image

    def predict_image(self, image):
        """
        Preprocess the image and make a prediction.
        """
        if self.model is None:
            raise ValueError("Model is not set")
        preprocessed_image = self.preprocess_image(image)
        prediction = self.model.predict(preprocessed_image)
        class_labels = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TUBERCULOSIS']
        pred_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        return f"Prediction: {pred_class} ({confidence:.2f}%)"

    def ui(self):
        """
        Create the Gradio UI for the image classification.
        """
        iface = gr.Interface(
            fn=self.predict_image,
            inputs=gr.Image(type="numpy", label="Upload Image"),
            outputs=gr.Textbox(label="Prediction")
        )
        iface.launch()

    