import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

image_path = "./test_data/normal/rimg0057small.jpg"
trainedModel = 'best.pt'

file_path = 'classes.txt'
with open(file_path, 'r') as file:
    rows = file.readlines()

rows = [row.strip() for row in rows]
num_classes = len(rows)

class TransferLearningCNN(nn.Module):
    def __init__(self, num_classes):
        super(TransferLearningCNN, self).__init__()
        self.mobilenet_v2 = models.mobilenet_v2()
        in_features = self.mobilenet_v2.classifier[1].in_features
        self.mobilenet_v2.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.mobilenet_v2(x)

model = TransferLearningCNN(num_classes)
model.load_state_dict(torch.load(trainedModel))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

input_image = Image.open(image_path).convert('RGB')
input_tensor = transform(input_image)
input_batch = input_tensor.unsqueeze(0)  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
input_batch = input_batch.to(device)

with torch.no_grad():
    output = model(input_batch)

probabilities = torch.nn.functional.softmax(output[0], dim=0)
predicted_class = torch.argmax(probabilities).item()
confidence_score = probabilities[predicted_class].item()
predicted_class = rows[predicted_class]

print(f"Predicted Class: {predicted_class}")
print(f"Confidence Score: {confidence_score:.4f}")
