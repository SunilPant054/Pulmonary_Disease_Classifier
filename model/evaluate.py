import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

class EvaluateNetwork:
    def __init__(self, model, test_data, history=None):
        self.model = model
        self.test_data = test_data
        self.history = history

    def evaluate(self):
        test_loss, test_accuracy = self.model.evaluate(self.test_data, verbose=2)
        print(f"Test Loss: {test_loss}")
        print(f"Test Accuracy: {test_accuracy}")

        # Getting predictions
        predictions = self.model.predict(self.test_data, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = self.test_data.classes
        class_labels = list(self.test_data.class_indices.keys())

        # Printing classification report
        report = classification_report(true_classes, predicted_classes, target_names=class_labels)
        print("Classification Report:")
        print(report)

        # Generating confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        print("Confusion Matrix:")
        print(cm)
        
        # Plotting confusion matrix
        self.plot_confusion_matrix(cm, class_labels)

        # Accuracy
        accuracy = accuracy_score(true_classes, predicted_classes)
        print(f"Accuracy: {accuracy}")

        # Precision
        precision = precision_score(true_classes, predicted_classes, average='weighted')
        print(f"Precision: {precision}")

        # Recall
        recall = recall_score(true_classes, predicted_classes, average='weighted')
        print(f"Recall: {recall}")

        # F1 Score
        f1 = f1_score(true_classes, predicted_classes, average='weighted')
        print(f"F1 Score: {f1}")

        # ROC-AUC Score
        roc_auc = roc_auc_score(true_classes, predictions, multi_class='ovr')
        print(f"ROC-AUC Score: {roc_auc}")

        # ROC Curve
        self.plot_roc_curve(true_classes, predictions, class_labels)

        return test_accuracy, report, cm, accuracy, precision, recall, f1, roc_auc

    def plot_confusion_matrix(self, cm, class_labels):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

    def plot_roc_curve(self, true_classes, predictions, class_labels):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(class_labels)):
            fpr[i], tpr[i], _ = roc_curve(true_classes == i, predictions[:, i])
            roc_auc[i] = roc_auc_score(true_classes == i, predictions[:, i])

        plt.figure()
        for i in range(len(class_labels)):
            plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {class_labels[i]} (area = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()
        
    def plot_training_history(self):
        if self.history is None:
            print("No training history data available")
            return
            
        #Plotting training & validation loss values
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        
        #Ploting training & validation accuracy values
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        
        plt.tight_layout()
        plt.show()
            
            
