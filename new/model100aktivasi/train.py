import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta
import json
import os

train_dir = 'C://Users//ASUS//Documents//tugasakhir//data_makanan-20240917T140626Z-001//data_makanan'
test_dir = 'C://Users//ASUS//Documents//tugasakhir//data_makanan-20240917T140626Z-001//test'

batch_size = 32
img_height = 224
img_width = 224

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

# Define different combinations of kernel sizes, activation functions, and optimizers
kernel_sizes = [(3, 3), (5, 5)]
activations = ['relu', 'tanh', 'sigmoid']
optimizers = [Adam, SGD, RMSprop, Adagrad, Adadelta]

# Create 15 combinations
combinations_parameter = [
    ((3, 3), 'relu', Adam),
    ((3, 3), 'tanh', SGD),
    ((3, 3), 'sigmoid', RMSprop),
    ((5, 5), 'relu', Adagrad),
    ((5, 5), 'tanh', Adadelta)
]

# Iterate over each combination
for idx, (kernel_size, activation, optimizer) in enumerate(combinations_parameter):
    print(f"Training model {idx+1}/{len(combinations_parameter)} with kernel_size={kernel_size}, activation={activation}, optimizer={optimizer.__name__}")

    model = Sequential([
        Conv2D(32, kernel_size, activation='relu', input_shape=(img_height, img_width, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(64, kernel_size, activation='tanh'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(128, kernel_size, activation='sigmoid'),
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

    epochs = 100

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=test_generator
    )

    model_path = f'C://Users//ASUS//Documents//tugasakhir//new//model100aktivasi//sign_language_cnn_model_{idx+1}.h5'
    model.save(model_path)

    # Save class labels
    class_labels = {v: k for k, v in train_generator.class_indices.items()}
    with open(f'C://Users//ASUS//Documents//tugasakhir//new//model100aktivasi//class_labels_model_{idx+1}.json', 'w') as f:
        json.dump(class_labels, f)

    # Save history
    history_path = f'C://Users//ASUS//Documents//tugasakhir//new//model100aktivasi//history_model_{idx+1}.json'
    with open(history_path, 'w') as f:
        json.dump(history.history, f)
