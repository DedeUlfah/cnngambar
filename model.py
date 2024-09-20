import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Path ke folder dataset gambar
folder_path = r'C:\Users\ASUS\Downloads\data_makanan-20240917T140626Z-001\data_makanan'

# Preprocessing data, scaling, dan splitting dataset (80% train, 20% validation)
datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=40,  # Kombinasi 3: Rotation lebih besar
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,  # Zoom range lebih agresif
    shear_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Kombinasi 3: Data split 80-20
)

# Generator untuk training data
train_generator = datagen.flow_from_directory(
    folder_path,
    target_size=(224, 224),  # Ukuran input untuk ResNet50 dan EfficientNetB0
    batch_size=32,  # Kombinasi 2: Batch size lebih besar
    class_mode='categorical',
    subset='training'  # Untuk data training
)

# Generator untuk validation data
valid_generator = datagen.flow_from_directory(
    folder_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # Untuk data validation
)

# Callback untuk early stopping dan learning rate scheduling
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

### Kombinasi Parameter 1: ResNet50, Unfreeze 5 Layer Terakhir ###
# Load ResNet50 pre-trained model tanpa lapisan fully connected terakhir (include_top=False)
base_model_resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Unfreeze the last 5 layers
for layer in base_model_resnet.layers[-5:]:
    layer.trainable = True

model_resnet = Sequential([
    base_model_resnet,
    Flatten(),
    Dense(512, activation='relu'),  # Neuron lebih banyak
    Dropout(0.),  # Dropout lebih rendah
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(23, activation='softmax')  # Sesuaikan dengan jumlah kelas
])

model_resnet.compile(optimizer=Adam(learning_rate=0.00001),  # Learning rate rendah
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

### Kombinasi Parameter 2: ResNet50, Learning Rate Scheduling dan Batch Size Lebih Besar ###
history_resnet = model_resnet.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=30,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=valid_generator.samples // valid_generator.batch_size,
    callbacks=[early_stopping, reduce_lr],  # Learning rate scheduling
    verbose=1
)

### Kombinasi Parameter 5: EfficientNetB0 dengan Learning Rate Rendah ###
# Load EfficientNetB0 pre-trained model tanpa lapisan fully connected terakhir (include_top=False)
base_model_effnet = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze semua layer EfficientNetB0
for layer in base_model_effnet.layers:
    layer.trainable = False

model_effnet = Sequential([
    base_model_effnet,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(23, activation='softmax')  # Sesuaikan dengan jumlah kelas
])

model_effnet.compile(optimizer=Adam(learning_rate=0.00001),  # Learning rate sangat rendah untuk fine-tuning
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

# Train model EfficientNetB0
history_effnet = model_effnet.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=30,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=valid_generator.samples // valid_generator.batch_size,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Prediksi menggunakan validation data untuk ResNet50
val_preds_resnet = model_resnet.predict(valid_generator)
val_preds_class_indices_resnet = np.argmax(val_preds_resnet, axis=1)

# Prediksi menggunakan validation data untuk EfficientNetB0
val_preds_effnet = model_effnet.predict(valid_generator)
val_preds_class_indices_effnet = np.argmax(val_preds_effnet, axis=1)

# True labels dari validation data
true_labels = valid_generator.classes

# Class labels
class_labels = list(valid_generator.class_indices.keys())

# Laporan klasifikasi untuk ResNet50
report_resnet = classification_report(true_labels, val_preds_class_indices_resnet, target_names=class_labels)
print("ResNet50 Classification Report:\n", report_resnet)

# Laporan klasifikasi untuk EfficientNetB0
report_effnet = classification_report(true_labels, val_preds_class_indices_effnet, target_names=class_labels)
print("EfficientNetB0 Classification Report:\n", report_effnet)

# Plot training & validation loss and accuracy untuk ResNet50
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history_resnet.history['loss'], label='Train Loss ResNet')
plt.plot(history_resnet.history['val_loss'], label='Validation Loss ResNet')
plt.legend()
plt.title('ResNet50 Loss')

plt.subplot(1, 2, 2)
plt.plot(history_resnet.history['accuracy'], label='Train Accuracy ResNet')
plt.plot(history_resnet.history['val_accuracy'], label='Validation Accuracy ResNet')
plt.legend()
plt.title('ResNet50 Accuracy')

plt.show()

# Plot training & validation loss and accuracy untuk EfficientNetB0
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history_effnet.history['loss'], label='Train Loss EfficientNet')
plt.plot(history_effnet.history['val_loss'], label='Validation Loss EfficientNet')
plt.legend()
plt.title('EfficientNetB0 Loss')

plt.subplot(1, 2, 2)
plt.plot(history_effnet.history['accuracy'], label='Train Accuracy EfficientNet')
plt.plot(history_effnet.history['val_accuracy'], label='Validation Accuracy EfficientNet')
plt.legend()
plt.title('EfficientNetB0 Accuracy')

plt.show()

# Save models
model_resnet.save('resnet50_model.h5')
model_effnet.save('efficientnetb0_model.h5')
