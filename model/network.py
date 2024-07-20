from keras import layers, Sequential
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.regularizers import l2

class ClassifierModel:

    def network(self):
        model = Sequential()
        
        model.add(layers.Conv2D(32,
                                kernel_size=(3, 3),
                                padding='valid',
                                activation='relu',
                                input_shape=(256, 256, 3),
                                kernel_regularizer=l2(0.0001)
                                ))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(2, 2))

        model.add(layers.Conv2D(64,
                                kernel_size=(3, 3),
                                padding='valid',
                                activation='relu',
                                kernel_regularizer=l2(0.0001)
                                ))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(2, 2))
        
        model.add(layers.Conv2D(128,
                                kernel_size=(3, 3),
                                padding='valid',
                                activation='relu',
                                kernel_regularizer=l2(0.0001)
                                ))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(2, 2))
        
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.5))  
        model.add(layers.Dense(256, activation='relu', kernel_regularizer=l2(0.0001)))
        model.add(layers.Dropout(0.5))  
        model.add(layers.Dense(4, activation='softmax'))

        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model