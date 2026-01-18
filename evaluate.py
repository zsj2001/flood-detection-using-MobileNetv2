import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

current_directory = os.getcwd()

image_folder_path = current_directory + "/test_data"
trained_model_path = 'best.pt'
class_file_path = 'classes.txt'

with open(class_file_path, 'r') as file:
    rows = file.readlines()

class_names = [row.strip() for row in rows]
num_classes = len(class_names)

class TransferLearningCNN(nn.Module):
    def __init__(self, num_classes):
        super(TransferLearningCNN, self).__init__()
        self.mobilenet_v2 = models.mobilenet_v2()
        in_features = self.mobilenet_v2.classifier[1].in_features
        self.mobilenet_v2.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.mobilenet_v2(x)

model = TransferLearningCNN(num_classes)
model.load_state_dict(torch.load(trained_model_path))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

testing_set = ImageFolder(image_folder_path, transform=transform)
testing_loader = DataLoader(testing_set, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

all_labels = []
all_predictions = []

with torch.no_grad():
    for inputs, labels in testing_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

conf_matrix = confusion_matrix(all_labels, all_predictions)
accuracy = accuracy_score(all_labels, all_predictions)
recall = recall_score(all_labels, all_predictions, average='binary')
precision = precision_score(all_labels, all_predictions, average='binary')

# Calculate True Negative Rate
tn, fp, fn, tp = conf_matrix.ravel()
true_negative_rate = tn / (tn + fp) if (tn + fp) != 0 else 0

print(conf_matrix.T)
print(f"Accuracy: {accuracy}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"True Negative Rate: {true_negative_rate}")


plt.figure(figsize=(num_classes, num_classes))
sns.heatmap(conf_matrix.T, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("True")
plt.ylabel("Predicted")
plt.title("Confusion Matrix")
plt.show()
