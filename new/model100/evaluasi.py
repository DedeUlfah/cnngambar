import os
import cv2
import numpy as np
from keras.models import load_model
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define constants
data_dir = r"C:\Users\Sinta\Downloads\data\data\data30"
image_size = (227, 227)
num_classes = len(os.listdir(data_dir))

# Load data
X = []
y = []

# Initialize a dictionary to store the count of images in each class
class_counts = {}

# Iterate through the categories
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
    
# Split the data into training and test sets (optional: this is just to keep the test set same as before)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding
y_test = label_binarize(y_test, classes=range(num_classes))

# Load the pre-trained model
model_path = r'C:\Users\Sinta\Documents\Documents\Dede Ulfah\cnngambar\new\model100\models\model_sign_language.h5'
model = load_model(model_path)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Plot Confusion Matrix
conf_matrix = metrics.confusion_matrix(np.argmax(y_test, axis=1), y_pred_classes)

fig, ax = plt.subplots(figsize=(10, 8))
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=os.listdir(data_dir))

# Display the confusion matrix without immediately showing the plot
cm_display.plot(ax=ax, cmap='Blues', xticks_rotation=45)

# Rotate x-axis labels to be diagonal (serong kanan)
plt.xticks(rotation=45, ha='right')  # ha='right' makes the labels aligned to the right
# plt.yticks(rotation=45, va='center')  # va='center' keeps y-axis labels centered

# Show the plot
plt.tight_layout()  # Ensures labels fit well within the plot
plt.show()

# Classification Report
print("\nClassification Report:\n")
print(metrics.classification_report(np.argmax(y_test, axis=1), y_pred_classes, target_names=os.listdir(data_dir)))

# Calculate Accuracy, Precision, Recall, F1 Score
accuracy = metrics.accuracy_score(np.argmax(y_test, axis=1), y_pred_classes)
precision = metrics.precision_score(np.argmax(y_test, axis=1), y_pred_classes, average='weighted')
recall = metrics.recall_score(np.argmax(y_test, axis=1), y_pred_classes, average='weighted')
f1 = metrics.f1_score(np.argmax(y_test, axis=1), y_pred_classes, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# ROC Curve and AUC for each class
fpr = {}
tpr = {}
roc_auc = {}

for i in range(num_classes):
    fpr[i], tpr[i], _ = metrics.roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])

# Plot ROC Curves
plt.figure(figsize=(10, 8))
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Each Class')
plt.legend(loc='lower right')
plt.show()

# Overall AUC score
y_test_bin = label_binarize(np.argmax(y_test, axis=1), classes=range(num_classes))
y_pred_bin = label_binarize(y_pred_classes, classes=range(num_classes))

# Compute overall AUC score
overall_auc = metrics.roc_auc_score(y_test_bin, y_pred_bin, multi_class='ovr')
print(f"Overall AUC score: {overall_auc}")
