from data.data_loader import DataLoader
from model.evaluate import EvaluateNetwork
from model.network import ClassifierModel
from model.predict import Predict
from model.train_network import TrainNetwork
from view.ui import Frontend
from keras._tf_keras.keras.models import load_model
import os


class Main:
    def __init__(self):
        self.base_dir = "/home/pneuma/Desktop/ML/Deep Learning/PulmonaryDiseaseClassifier/dataset/"
        self.model_path = "/home/pneuma/Desktop/ML/Deep Learning/PulmonaryDiseaseClassifier/model.h5"
        self.data_loader = DataLoader(
            self.base_dir, val_split=0.2, batch_size=32)
        print("Loading data...")
        self.train, self.val, self.test = self.data_loader.load_data()
        print("Data loaded successfully.")
        self.model = None
        self.predict = None
        self.evaluate = None
        self.history = None

    def load_or_train_model(self):
        """Load the model if it exists, otherwise create, train, and save a new model."""
        if os.path.exists(self.model_path):
            try:
                # Load the saved model
                self.model = load_model(self.model_path)
                self.model.summary()
                print("Model loaded successfully.")
            except Exception as e:
                print(f"Error loading the model: {
                        e}. A new model will be created and trained.")
                self.create_and_train_model()
        else:
            print("Model does not exist. A new model will be created and trained.")
            self.create_and_train_model()

    def create_and_train_model(self):
        """Create a new model, train it, and save it."""
        m1 = ClassifierModel()
        self.model = m1.network()
        fit = TrainNetwork(self.train, self.val)
        self.model, self.history = fit.train_model(self.model)
        self.model.save(self.model_path)

    def evaluate_model(self):
        """Evaluate the model and print the results."""
        self.evaluate = EvaluateNetwork(self.model, self.test, self.history)
        self.evaluate.plot_training_history()
        test_accuracy, report, cm, accuracy, precision, recall, f1, roc_auc = self.evaluate.evaluate()
        print(f"Test Accuracy: {test_accuracy:.2f}")
        print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {
                recall:.2f}, F1 Score: {f1:.2f}, ROC AUC: {roc_auc:.2f}")
        print("Classification Report:")
        print(report)
        print("Confusion Matrix:")
        print(cm)

    def make_predictions(self):
        """Make predictions using the model and print the results."""
        self.predict = Predict(self.model, self.test)
        predictions, predicted_classes = self.predict.predict_model()
        print("Sample Predictions:")
        for pred in predictions[:5]:
            print(pred)  # Adjust the number of samples to show as needed
        print("Sample Predicted Classes:")
        for cls in predicted_classes[:5]:
            print(cls)  # Adjust the number of samples to show as needed

    def start_ui(self):
        """Start the user interface for image classification."""
        view = Frontend()
        view.set_model(self.model)  # Set the model for the Frontend
        view.ui()

    def menu(self):
        """Display a terminal menu for the user to choose an action."""
        while True:
            print("\n*** Pulmonary Disease Classification Menu ***")
            print("1. Load or Train Model")
            print("2. Predict on Test Data")
            print("3. Evaluate Model")
            print("4. Start User Interface")
            print("5. Exit")
            choice = input("Enter your choice (1-5): ")

            if choice == '1':
                self.load_or_train_model()
            elif choice == '2':
                self.make_predictions()
            elif choice == '3':
                self.evaluate_model()
            elif choice == '4':
                self.start_ui()
            elif choice == '5':
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please enter a number between 1 and 5.")


if __name__ == "__main__":
    try:
        print("Starting program...")
        main_instance = Main()
        main_instance.menu()
        print("Program ended.")
    except Exception as e:
        print(f"Error in main: {e}")
