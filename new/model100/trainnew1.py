import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch
import matplotlib.pyplot as plt
import json
import os

# Preprocessing seperti sebelumnya
train_dir = r"C:\Users\LABKOM2\Downloads\data\data30"
test_dir = r"C:\Users\LABKOM2\Downloads\data\data30 test"

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

# Define the HyperModel
class ResNetHyperModel(HyperModel):
    def build(self, hp):
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
        base_model.trainable = False  # Transfer learning, freeze ResNet layers

        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            BatchNormalization(),
            Dense(hp.Int('units', min_value=64, max_value=512, step=64), 
                  activation=hp.Choice('activation', values=['relu', 'tanh', 'sigmoid'])),
            Dropout(hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)),
            Dense(num_classes, activation='softmax')
        ])

        # Choose an optimizer
        optimizer = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop'])

        if optimizer == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG'))
        elif optimizer == 'sgd':
            opt = tf.keras.optimizers.SGD(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG'), momentum=hp.Float('momentum', min_value=0.0, max_value=0.9, step=0.1))
        elif optimizer == 'rmsprop':
            opt = tf.keras.optimizers.RMSprop(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG'))

        model.compile(optimizer=opt, 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])
        return model

# Hyperparameter Tuning
tuner = RandomSearch(
    ResNetHyperModel(),
    objective='val_accuracy',
    max_trials=10,  # Jumlah percobaan untuk kombinasi hyperparameter
    executions_per_trial=1,  # Eksekusi per trial
    directory='resnet50_tuning',  # Folder penyimpanan
    project_name='sign_language_tuning'
)

# Callback untuk menghentikan training jika tidak ada improvement
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Mulai proses tuning
tuner.search(train_generator, 
             epochs=20, 
             validation_data=test_generator, 
             callbacks=[stop_early])

# Mendapatkan hyperparameters terbaik
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
Best Hyperparameters:
- Units: {best_hps.get('units')}
- Activation: {best_hps.get('activation')}
- Optimizer: {best_hps.get('optimizer')}
- Learning Rate: {best_hps.get('learning_rate')}
""")

# Setelah menemukan hyperparameter terbaik, lakukan pelatihan ulang
model = tuner.hypermodel.build(best_hps)

history = model.fit(
    train_generator,
    epochs=20,
    validation_data=test_generator
)

# Save the best model
model.save(r'C:\Users\LABKOM2\cnngambar\new\models1\best_sign_language_model.h5')

# Plotting accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(20)

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

plt.show()
