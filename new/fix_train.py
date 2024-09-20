import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta
import json
import os

train_dir = 'C://Users//Sinta//Documents//Tugas Akhir//Backup Plan//sign_language_detection//dataset//train'
test_dir = 'C://Users//Sinta//Documents//Tugas Akhir//Backup Plan//sign_language_detection//dataset//test'

batch_size = 32
img_height = 64
img_width = 64

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
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

num_classes = len(train_generator.class_indices)

# Define the best combination of kernel size, activation function, and optimizer for model 10
kernel_size = (5, 5)
activation = 'relu'
optimizer = Adam

# Create the model using the best combination
model = Sequential([
    Conv2D(32, kernel_size, activation=activation, input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, kernel_size, activation=activation),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, kernel_size, activation=activation),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(512, activation=activation),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=optimizer(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

epochs = 50

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=test_generator
)

model_path = 'C://Users//Sinta//Documents//Tugas Akhir//Backup Plan//sign_language_detection//scripts//new//models//sign_language_cnn_model_best.h5'
model.save(model_path)

# Save class labels
class_labels = {v: k for k, v in train_generator.class_indices.items()}
with open('C://Users//Sinta//Documents//Tugas Akhir//Backup Plan//sign_language_detection//scripts//new//class_labels_model_best.json', 'w') as f:
    json.dump(class_labels, f)

# Save training history
history_path = 'C://Users//Sinta//Documents//Tugas Akhir//Backup Plan//sign_language_detection//scripts//new//history_model_best.json'
with open(history_path, 'w') as f:
    json.dump(history.history, f)

print('Training completed and model saved.')
