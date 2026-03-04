import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras import layers, models

# Direct loading - NO generators
train_ds = image_dataset_from_directory(
    'data/Combined Dataset/train', image_size=(128,128), batch_size=32, color_mode='grayscale'
)
test_ds = image_dataset_from_directory(
    'data/Combined Dataset/test', image_size=(128,128), batch_size=32, color_mode='grayscale'
)

model = models.Sequential([
    layers.Conv2D(32, 5, activation='relu', input_shape=(128,128,1)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, 5, activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, epochs=10, validation_data=test_ds)
model.save('models/lenet_raw.h5')
