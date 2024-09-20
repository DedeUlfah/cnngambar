import os
import cv2
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from keras.applications import ResNet50
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.optimizers import Adam
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Define constants
data_dir = r"C:\Users\LABKOM2\Downloads\data\data30"
output_dir = r"C:\Users\LABKOM2\Downloads\data\data30 test"
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

# Create the model
base_model = ResNet50(weights=None, include_top=False, input_shape=(227, 227, 3))

model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

start_time = time.time()

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50)

# Calculate training time
training_time = time.time() - start_time
print("\nTraining time: {:.2f} seconds".format(training_time))

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

# Plot training loss
plt.plot(train_loss, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Training Loss')
plt.legend()
plt.show()

# Plot training accuracy
plt.plot(train_accuracy, label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Training Accuracy')
plt.legend()
plt.show()


# Evaluate the model on the test set
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

# Calculate confusion matrix and other metrics
confusion_matrix = metrics.confusion_matrix(np.argmax(y_test, axis=1), y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=os.listdir(data_dir))
cm_display.plot()
plt.show()

accuracy = metrics.accuracy_score(np.argmax(y_test, axis=1), y_pred)
precision = metrics.precision_score(np.argmax(y_test, axis=1), y_pred, average='weighted')
recall = metrics.recall_score(np.argmax(y_test, axis=1), y_pred, average='weighted')
f1_score = metrics.f1_score(np.argmax(y_test, axis=1), y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
print(classification_report(np.argmax(y_test, axis=1), y_pred, target_names=os.listdir(data_dir)))

# Define class names
class_names = ['ayam goreng', 'ayam pop', 'gulai tambusu', 'kue ape', 'kue bika ambon', 'kue cenil', 'kue dadar gulung', 'kue gethuk lidri', 'kue kastangel', 'kue klepon', 'kue lapis', 'kue lumpur', 'kue nagasari', 'kue pastel', 'kue putri salju', 'kue risoles', 'lemper', 'lumpia', 'putu ayu', 'serabi solo', 'telur balado', 'telur dadar', 'wajik']

# Convert y_test and y_pred to one-hot encoded format (if not already)
y_test_bin = label_binarize(y_test, classes=range(num_classes))
y_pred_bin = label_binarize(y_pred, classes=range(num_classes))

# Calculate ROC and AUC for each class
fpr = {}
tpr = {}
roc_auc = {}

for i in range(num_classes):
    fpr[i], tpr[i], _ = metrics.roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])

# Plot ROC curves for each class
plt.figure(figsize=(8, 6))
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label='ROC curve ({}) - AUC = {:.2f}'.format(class_names[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

# Convert y_test and y_pred to one-hot encoded format (if not already)
y_test_bin = label_binarize(y_test, classes=range(num_classes))
y_pred_bin = label_binarize(y_pred, classes=range(num_classes))

# Calculate overall AUC using 'ovr' strategy
auc_score = roc_auc_score(y_test_bin, y_pred_bin, multi_class='ovr')

# Print the overall AUC score
print("Overall AUC score:", auc_score)