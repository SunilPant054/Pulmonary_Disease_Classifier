import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    def __init__(self, history) -> None:
        self.history = history
        
    def plot_training_history(self):
        """
        Plot the training and validation accuracy/loss curves.
        """
        if self.history is None:
            raise ValueError("Training history is not set")
        
        plt.figure(figsize=(12, 5))
        
        #Plot trainign and validation accuracy values
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Validation'])
        
        #Plot training & validaiton loss values 
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'])
        
        plt.show()
        
    def confusion_matrix(self, conf_matrix, class_names):
        """
        Plot the confusion matrix as a heatmap
        """
        
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()