import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
import json
import os
import matplotlib.pyplot as plt

train_dir = r"C:\Users\LABKOM2\Downloads\data\data30"
test_dir = r"C:\Users\LABKOM2\Downloads\data\data30 test"

batch_size = 64
img_height = 64
img_width = 64

# Data augmentation dan preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

# Load ResNet50 sebagai backbone
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Membekukan sebagian besar layer di ResNet50
for layer in base_model.layers[:-10]:  # Biarkan 10 layer terakhir untuk trainable
    layer.trainable = True

# Definisikan arsitektur CNN
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1024, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer=Adam(learning_rate=1e-5),  # Cobalah learning rate lebih kecil
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Summary model
model.summary()

# Callbacks (misalnya early stopping dan pengurangan learning rate saat terjebak)
my_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)
]

epochs = 20

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=test_generator,
    validation_steps=len(test_generator),
    epochs=epochs,
    callbacks=my_callbacks
)

# Save model
model_path = f'C:/Users/LABKOM2/cnngambar/new/models1/sign_language_cnn_resnet50_model_v3.h5'
model.save(model_path)

# Save class labels
class_labels = {v: k for k, v in train_generator.class_indices.items()}
with open(f'C:/Users/LABKOM2/cnngambar/new/models1/class/class_labels_resnet50_v3.json', 'w') as f:
    json.dump(class_labels, f)

# Save training history
history_path = f'C:/Users/LABKOM2/cnngambar/new/models1/history/history_resnet50_v3.json'
with open(history_path, 'w') as f:
    json.dump(history.history, f)

# Plotting accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

plt.figure(figsize=(12, 6))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

# Save the plots
plot_path = f'C:/Users/LABKOM2/cnngambar/new/models1/plot/plot_resnet50_v3.png'
plt.savefig(plot_path)

# Display the plots
plt.show()
