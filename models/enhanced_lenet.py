import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt

# Same data
train_ds = image_dataset_from_directory('data/Combined Dataset/train', image_size=(128,128), batch_size=32, label_mode='int', color_mode='grayscale')
test_ds = image_dataset_from_directory('data/Combined Dataset/test', image_size=(128,128), batch_size=32, label_mode='int', color_mode='grayscale')

# **ENHANCED LeNet** - Balanced architecture (your abstract)
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(128, 128, 1)),
    
    # Layer 1: Original LeNet + light enhancement
    layers.Conv2D(32, 5, activation='relu', padding='same'),
    layers.MaxPooling2D(2, strides=2),  # Your "improved MaxPooling2D"
    
    # Layer 2: Enhanced depth
    layers.Conv2D(64, 5, activation='relu', padding='same'),
    layers.Dropout(0.25),  # Your Dropout innovation
    layers.MaxPooling2D(2, strides=2),
    
    # Layer 3: Extension (your contribution)
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(2, strides=1, padding='same'),
    
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(4, activation='softmax')
])

# Lower learning rate for stability
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

print("📈 Enhanced LeNet v2 (Fixed):")
model.summary()

# Better callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

history = model.fit(train_ds, epochs=25, validation_data=test_ds, callbacks=callbacks, verbose=1)

test_loss, test_acc = model.evaluate(test_ds)
print(f"\n🎯 ENHANCED LeNet Test Accuracy: {test_acc:.2%}")

model.save('models/enhanced_lenet_v2.keras')  # Modern format
plt.figure(figsize=(15,5))
plt.subplot(1,3,1); plt.plot(history.history['accuracy'], label='Train'); plt.plot(history.history['val_accuracy'], label='Val'); plt.title('Enhanced LeNet Accuracy'); plt.legend()
plt.subplot(1,3,2); plt.plot(history.history['loss'], label='Train'); plt.plot(history.history['val_loss'], label='Val'); plt.title('Loss'); plt.legend()
plt.subplot(1,3,3); plt.imshow(next(iter(test_ds))[0][0][:,:,0], cmap='gray'); plt.title('Sample MRI Slice'); plt.axis('off')
plt.savefig('models/enhanced_results.png')
plt.show()
