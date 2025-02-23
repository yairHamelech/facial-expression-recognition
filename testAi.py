import torch
from PIL import Image
from torchvision import transforms
from torch import nn
import timm
import matplotlib.pyplot as plt


class FaceModel(nn.Module):
    def __init__(self):
        super(FaceModel, self).__init__()
        # Use efficientnet_b3 as base model
        self.eff_net = timm.create_model('efficientnet_b3', pretrained=True, num_classes=7)

        # Modify the final layer (classifier) to have more neurons
        self.eff_net.classifier = nn.Sequential(
            nn.Linear(self.eff_net.classifier.in_features, 2048),  # Increase to 2048 neurons in hidden layer
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout to reduce overfitting
            nn.Linear(2048, 7)  # Output layer (7 classes)
        )

    def forward(self, images, labels=None):
        logits = self.eff_net(images)
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return logits, loss
        return logits


# מוריד את המודל
model = FaceModel()
model.load_state_dict(torch.load("models_best_weights.pt", weights_only=True))

model.eval()

# משנה את התמונה שתתאים למודל
transform = transforms.Compose([

    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img_path = r'C:\Users\yair2\PycharmProjects\facial-expression-recognition\istockphoto-184600247-612x612.jpg'
image = Image.open(img_path)

img_tensor = transform(image).unsqueeze(0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
img_tensor = img_tensor.to(device)

# נותן חיזוי
with torch.no_grad():
    logits = model(img_tensor)
    probabilities = torch.softmax(logits, dim=1)


classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
predicted_class = classes[probabilities.argmax()]
print(f"Predicted Emotion: {predicted_class}")

# פונקצייה ויזואלית (משהו זמני רק כדי להראות את החיזוי בצורה יפה)
def view_classify(img, ps):
    classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    ps = ps.data.cpu().numpy().squeeze()
    img = img.cpu().numpy().transpose(1, 2, 0)  # Ensure img is on CPU

    fig, (ax1, ax2) = plt.subplots(figsize=(5, 9), ncols=2)
    ax1.imshow(img)
    ax1.axis('off')
    ax2.barh(classes, ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(classes)
    ax2.set_yticklabels(classes)
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()
    plt.show()


view_classify(img_tensor.squeeze(0), probabilities)
