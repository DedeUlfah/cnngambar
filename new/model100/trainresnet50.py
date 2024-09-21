import os
import cv2
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.applications import ResNet50
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# Define constants
data_dir = r"C:\Users\Sinta\Downloads\data\data\data30"
output_dir = r"C:\Users\Sinta\Downloads\data\data\data30 test"
image_size = (227, 227)
num_classes = len(os.listdir(data_dir))

# Load data
X = []
y = []

# Initialize a dictionary to store the count of images in each class
class_counts = {}

# Iterate through the flower categories
for category in os.listdir(data_dir):
    category_dir = os.path.join(data_dir, category)
    class_counts[category] = len(os.listdir(category_dir))
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

# Print the count of images in each class
for category, count in class_counts.items():
    print(f"Class: {category}, Count: {count}")
    
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train dataset shape:", X_train.shape)
print("Train labels shape:", y_train.shape)
print("Test dataset shape:", X_test.shape)
print("Test labels shape:", y_test.shape)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

datagen.fit(X_train)

# Transfer Learning - Use pre-trained ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(227, 227, 3))

# Freeze base_model layers
for layer in base_model.layers:
    layer.trainable = False

# Create the model
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(1024, activation='relu', kernel_regularizer='l2'))  # L2 Regularization
model.add(BatchNormalization())  # Batch normalization
model.add(Dropout(0.5))  # Dropout
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for early stopping and saving best model
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint(filepath='best_model.h5', monitor='val_accuracy', save_best_only=True)
]

start_time = time.time()

# Train model with data augmentation
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    validation_data=(X_test, y_test),
                    epochs=5, callbacks=callbacks)

# Calculate training time
training_time = time.time() - start_time
print("\nTraining time: {:.2f} seconds".format(training_time))

# Save the trained model
model_path = r'C:\Users\Sinta\Documents\Documents\Dede Ulfah\cnngambar\new\model100\models\model_sign_language.h5'
model.save(model_path)

# Get training history
train_loss = history.history['loss']
train_accuracy = history.history['accuracy']

# Plot training loss and training accuracy
plt.plot(train_loss, label='Training Loss')
plt.plot(train_accuracy, label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Training Loss and Accuracy')
plt.legend()
plt.show()

# Evaluate the model on the test set
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

# Calculate confusion matrix and display
confusion_mtx = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_mtx, display_labels=os.listdir(data_dir))
cm_display.plot()
plt.show()

# Print classification metrics
accuracy = np.mean(np.argmax(y_test, axis=1) == y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(np.argmax(y_test, axis=1), y_pred, target_names=os.listdir(data_dir)))

# ROC Curve and AUC Calculation
from sklearn.preprocessing import label_binarize

y_test_bin = label_binarize(np.argmax(y_test, axis=1), classes=range(num_classes))
y_pred_bin = label_binarize(y_pred, classes=range(num_classes))

fpr = {}
tpr = {}
roc_auc = {}

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_auc_score(y_test_bin[:, i], y_pred_bin[:, i])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])

# Plot ROC curve
plt.figure(figsize=(8, 6))
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve ({i}) - AUC = {roc_auc[i]:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

# Overall AUC
auc_score = roc_auc_score(y_test_bin, y_pred_bin, multi_class='ovr')
print("Overall AUC score:", auc_score)
