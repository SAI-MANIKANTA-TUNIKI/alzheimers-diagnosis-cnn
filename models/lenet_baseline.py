import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

TRAIN_DIR = 'data/Combined Dataset/train'
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10  # Reduced for quick test

# FIXED: No validation_split, use test set instead + grayscale fix
datagen = ImageDataGenerator(
    rescale=1./255, rotation_range=5, width_shift_range=0.05,
    height_shift_range=0.05, zoom_range=0.05
)

# Load ALL train data (no split)
train_gen = datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', color_mode='grayscale', shuffle=True
)

# Test set (no augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    'data/Combined Dataset/test',  # ← FIXED PATH
    target_size=IMG_SIZE, 
    batch_size=BATCH_SIZE,
    class_mode='categorical', 
    color_mode='grayscale', 
    shuffle=False
)


print("✅ Train classes:", train_gen.class_indices)
print("✅ Test classes:", test_gen.class_indices)

# LeNet-5 for grayscale (128x128x1)
model = models.Sequential([
    layers.Conv2D(32, (5,5), activation='relu', input_shape=(128,128,1)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (5,5), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(4, activation='softmax')  # 4 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train on full train set
print("🚀 Training LeNet...")
history = model.fit(train_gen, epochs=EPOCHS, validation_data=test_gen)

# Final test eval
test_loss, test_acc = model.evaluate(test_gen)
print(f"✅ FINAL LeNet Test Accuracy: {test_acc:.2%}")

model.save('models/lenet_baseline.h5')

# Plot
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Test')
plt.title('LeNet Accuracy'); plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Test')
plt.title('LeNet Loss'); plt.legend()
plt.savefig('models/lenet_results.png')
plt.show()
