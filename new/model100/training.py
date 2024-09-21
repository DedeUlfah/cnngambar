import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras_tuner import RandomSearch
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report

# Define constants
data_dir = r"C:\Users\Sinta\Downloads\data\data\data30"
image_size = (227, 227)

# Load data
X = []
y = []

# Iterate through the categories in one folder
for category in os.listdir(data_dir):
    category_dir = os.path.join(data_dir, category)
    for image_file in os.listdir(category_dir):
        image_path = os.path.join(category_dir, image_file)
        image = cv2.imread(image_path)
        image = cv2.resize(image, image_size)
        X.append(image)
        y.append(category)

X = np.array(X)
y = np.array(y)

# Encode class labels to numerical values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding
num_classes = len(np.unique(y))
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Build the model using Keras Tuner
def build_model(hp):
    base_model = ResNet50(weights=None, include_top=False, input_shape=(227, 227, 3))

    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    
    # Tuning dense layer size
    model.add(Dense(units=hp.Int('units', min_value=512, max_value=2048, step=512), activation='relu'))
    
    # Tuning dropout rate
    model.add(Dropout(rate=hp.Float('dropout', min_value=0.3, max_value=0.6, step=0.1)))
    
    model.add(Dense(num_classes, activation='softmax'))

    # Tuning learning rate
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, sampling='log')

    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

# Set up the tuner
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=1,
    directory='tuner_results',
    project_name='sign_language_detection'
)

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Start searching for the best hyperparameters
start_time = time.time()

tuner.search(X_train, y_train, validation_data=(X_test, y_test), epochs=10, callbacks=[early_stopping])

# Get the best model from the tuner
best_model = tuner.get_best_models(num_models=1)[0]

# Calculate training time
training_time = time.time() - start_time
print("\nTraining time: {:.2f} seconds".format(training_time))

# Save the best model
model_path = r'C:\Users\Sinta\Documents\Documents\Dede Ulfah\cnngambar\new\model100\models\model_sign_language_best.h5'
best_model.save(model_path)

# Evaluate the best model on the test set
test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

# Predict on the test set
y_pred = best_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calculate confusion matrix and classification report
conf_matrix = confusion_matrix(y_true, y_pred_classes)
print(conf_matrix)

# Display confusion matrix
plt.figure(figsize=(10, 8))
plt.title("Confusion Matrix")
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Print classification report
print(classification_report(y_true, y_pred_classes, target_names=label_encoder.classes_))

# Visualize training loss and accuracy history
history = best_model.history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Training Loss and Accuracy')
plt.legend()
plt.show()
