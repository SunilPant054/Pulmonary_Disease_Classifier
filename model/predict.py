import numpy as np

class Predict:
    def __init__(self, model, test_data):
        self.model = model
        self.test_data = test_data

    def predict_model(self):
        predictions = self.model.predict(self.test_data, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        return predictions, predicted_classes