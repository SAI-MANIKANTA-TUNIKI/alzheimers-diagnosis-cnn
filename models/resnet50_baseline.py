import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.applications import ResNet50
import matplotlib.pyplot as plt

# Data (your perfect setup)
train_ds = image_dataset_from_directory('data/Combined Dataset/train', image_size=(224,224), batch_size=32, label_mode='int')
test_ds = image_dataset_from_directory('data/Combined Dataset/test', image_size=(224,224), batch_size=32, label_mode='int')

# ResNet50 Transfer Learning
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False  # Freeze for baseline

model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(128, activation='relu'),
    layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("ResNet50 Baseline:")
model.summary()

history = model.fit(train_ds, epochs=8, validation_data=test_ds)

test_loss, test_acc = model.evaluate(test_ds)
print(f"🎯 ResNet50 Test Accuracy: {test_acc:.2%}")
model.save('models/resnet50_baseline.keras')
