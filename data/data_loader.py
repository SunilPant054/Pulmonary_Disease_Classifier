import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

class DataLoader:

    def __init__(self, base_dir, val_split=0.2, batch_size=32) -> None:
        self.base_dir = base_dir
        self.val_split = val_split  # Fraction of data to be used for validation
        self.batch_size = batch_size

    def load_data(self):
        # Normalizing image in the range of 0 and 1
        # Create the ImageDataGenerator for train data with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest',
            shear_range=0.1,
            brightness_range=[0.9, 1.1],
            validation_split=self.val_split  # Specify the split fraction for validation
        )

        # Training data generator with augmentation
        train_generator = train_datagen.flow_from_directory(
            directory=self.base_dir + 'train',
            target_size=(256, 256),
            batch_size=self.batch_size,
            shuffle=True,
            class_mode='categorical',
            subset='training'  # Training data
        )

        # Validation data generator without augmentation
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=self.val_split  # Specify the split fraction for validation
        )

        val_generator = val_datagen.flow_from_directory(
            directory=self.base_dir + 'train',
            target_size=(256, 256),
            batch_size=self.batch_size,
            shuffle=False,
            class_mode='categorical',
            subset='validation'  # Validation data
        )

        # Test data generator without augmentation
        test_datagen = ImageDataGenerator(
            rescale=1./255
        )

        test_generator = test_datagen.flow_from_directory(
            directory=self.base_dir + 'test',
            target_size=(256, 256),
            batch_size=self.batch_size,
            shuffle=False,  # Do not shuffle test data
            class_mode='categorical'
        )

        return train_generator, val_generator, test_generator
