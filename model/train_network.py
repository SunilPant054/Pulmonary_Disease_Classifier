from keras._tf_keras.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

import math

import tensorflow as tf

class TrainNetwork:
    def __init__(self, train_data, val_data):
        self.train_data = train_data
        self.val_data = val_data

    def train_model(self, model):
        # Callbacks for learning rate reduction and early stopping
        lr_reduction = ReduceLROnPlateau(
            monitor='val_loss', 
            patience=2, 
            verbose=1, 
            factor=0.5, 
            min_lr=1e-6
        )

        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=3, 
            verbose=1, 
            restore_best_weights=True
        )

        def scheduler(epoch, lr):
            if epoch < 10:
                return lr
            else:
                return float(lr * tf.math.exp(-0.1))
        
        lr_scheduler = LearningRateScheduler(scheduler)

        history = model.fit(
            self.train_data,
            epochs=20,
            validation_data=self.val_data,
            shuffle=True,
            callbacks=[lr_reduction, early_stopping, lr_scheduler]
        )
        return model, history
