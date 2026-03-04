import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
import numpy as np

# Direct load from YOUR raw folders - NO PREPROCESSING NEEDED
print("🔄 Loading raw data...")
train_ds = image_dataset_from_directory(
    'data/Combined Dataset/train',
    image_size=(128, 128),
    batch_size=32,
    label_mode='int',  # Integer labels for sparse_categorical_crossentropy
    color_mode='grayscale'
)

test_ds = image_dataset_from_directory(
    'data/Combined Dataset/test',
    image_size=(128, 128),
    batch_size=32,
    label_mode='int',
    color_mode='grayscale'
)

print("✅ Train classes:", train_ds.class_names)
print("✅ Test classes:", test_ds.class_names)

# LeNet-5 Baseline (your abstract's 97.19% target)
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(128, 128, 1)),  # Normalize here
    layers.Conv2D(32, 5, activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, 5, activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(4, activation='softmax')  # 4 AD stages
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("📊 Model summary:")
model.summary()

# Train (10 epochs = ~10 mins)
print("🚀 Training LeNet Baseline...")
history = model.fit(
    train_ds, 
    epochs=10, 
    validation_data=test_ds,
    verbose=1
)

# Results
test_loss, test_acc = model.evaluate(test_ds)
print(f"\n🎯 FINAL LeNet Test Accuracy: {test_acc:.2%}")

model.save('models/lenet_baseline.h5')

# Plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Test Acc')
plt.title('LeNet Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('LeNet Loss')
plt.legend()
plt.savefig('models/lenet_results.png')
plt.show()

print("✅ SAVED: models/lenet_baseline.h5 + lenet_results.png")
