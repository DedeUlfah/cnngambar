import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta
from tensorflow.keras.applications import ResNet50
import json
import os
import matplotlib.pyplot as plt

train_dir = r"C:\Users\LABKOM2\Downloads\data\data30"
test_dir = r"C:\Users\LABKOM2\Downloads\data\data30 test"

batch_size = 32
img_height = 224  # Disarankan untuk ResNet50
img_width = 224   # Disarankan untuk ResNet50

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
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

num_classes = len(train_generator.class_indices)

# Kombinasi activation functions dan optimizers
combinations_parameter = [
    ('relu', Adam),
    ('relu', SGD),
    ('relu', RMSprop),
    ('tanh', Adam),
    ('tanh', SGD),
    ('tanh', RMSprop),
    ('sigmoid', Adam),
    ('sigmoid', SGD),
    ('sigmoid', RMSprop),
    ('sigmoid', Adagrad)
]

# Iterasi untuk setiap kombinasi hyperparameter
for idx, (activation, optimizer) in enumerate(combinations_parameter):
    print(f"Training model {idx+1}/{len(combinations_parameter)} with activation={activation}, optimizer={optimizer.__name__}")

    # Load model ResNet50 tanpa top layer (untuk transfer learning)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    # Freeze base model
    base_model.trainable = False

    # Definisikan arsitektur CNN dengan backbone ResNet50
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(128, activation=activation),  # Dense layer dengan activation function
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # Output layer
    ])

    # Compile model dengan optimizer yang dipilih
    model.compile(optimizer=optimizer(learning_rate=0.01),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    epochs = 20

    # Train the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=test_generator
    )

    # Save model
    model_path = f'C:/Users/LABKOM2/cnngambar/new/models1/sign_language_cnn_model_{idx+1}.h5'
    model.save(model_path)

    # Save class labels
    class_labels = {v: k for k, v in train_generator.class_indices.items()}
    with open(f'C:/Users/LABKOM2/cnngambar/new/models1/class/class_labels_model_{idx+1}.json', 'w') as f:
        json.dump(class_labels, f)

    # Save training history
    history_path = f'C:/Users/LABKOM2/cnngambar/new/models1/history/history_model_{idx+1}.json'
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
    plt.title(f'Model {idx+1}: Training and Validation Accuracy')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f'Model {idx+1}: Training and Validation Loss')

    # Save the plots
    plot_path = f'C:/Users/LABKOM2/cnngambar/new/models1/plot/plot_model_{idx+1}.png'
    plt.savefig(plot_path)

    # Display the plots
    plt.show()
