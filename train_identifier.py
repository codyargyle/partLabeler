from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from identifier_CNN import create_model

# Load and preprocess data
datagen = ImageDataGenerator(rescale=1.0/255.0)
train_data = datagen.flow_from_directory('data/train', target_size=(150, 150), batch_size=32, class_mode='sparse')

# Create and compile model
model = create_model((150, 150, 3), len(train_data.class_indices))
model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

# Train the model
model.fit(train_data, epochs=10)