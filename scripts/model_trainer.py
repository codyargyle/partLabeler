import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from data_loader import create_dataframe, create_dataset
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(4)  # Output layer for bounding box regression (xmin, ymin, xmax, ymax)
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
    return model

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.1

def create_augmented_dataset(dataframe, img_directory, batch_size):
    datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    try:
        generator = datagen.flow_from_dataframe(
            dataframe,
            directory=img_directory,
            x_col='filename',
            y_col=['xmin', 'ymin', 'xmax', 'ymax'],
            target_size=(150, 150),
            batch_size=batch_size,
            class_mode='raw',
            shuffle=True
        )
    except KeyError as e:
        print(f'KeyError: {e}. Check the column names in the DataFrame.')
        return None
    except Exception as e:
        print(f'An error occurred: {e}')
        return None
    
    return generator

def main():
    train_dir = '/home/cody/Documents/codyCodes/partLabeler/data/train'
    val_dir = '/home/cody/Documents/codyCodes/partLabeler/data/val'
    model_save_dir = '/home/cody/Documents/codyCodes/partLabeler/models'
    batch_size = 32
    input_shape = (150, 150, 3)
    
    print("Creating training dataset...")
    train_images_dir = os.path.join(train_dir, 'images')
    train_df = create_dataframe(train_dir, train_images_dir)
    train_dataset = create_augmented_dataset(train_df, train_images_dir, batch_size)
    
    print("Creating validation dataset...")
    val_images_dir = os.path.join(val_dir, 'images')
    val_df = create_dataframe(val_dir, val_images_dir)
    val_dataset = create_augmented_dataset(val_df, val_images_dir, batch_size)
    
    if train_dataset is None or val_dataset is None:
        print("Datasets could not be created. Check the dataset paths and files.")
        return
    
    model = build_model(input_shape)

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(os.path.join(model_save_dir, 'best_model.keras'), save_best_only=True, save_weights_only=False)
    lr_scheduler = LearningRateScheduler(scheduler)

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=50,  # Increased number of epochs
        callbacks=[early_stopping, model_checkpoint, lr_scheduler]
    )
    
    # Save the final model
    final_model_path = os.path.join(model_save_dir, 'model.keras')
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")

    # Print model summary
    print(model.summary())

    # Plot training history
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    main()
