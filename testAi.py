import torch
from PIL import Image
from torchvision import transforms
from torch import nn
import timm
import matplotlib.pyplot as plt


class FaceModel(nn.Module):
    def __init__(self):
        super(FaceModel, self).__init__()
        self.eff_net = timm.create_model('efficientnet_b0', pretrained=False, num_classes=7)

    def forward(self, images):
        logits = self.eff_net(images)
        return logits

# מוריד את המודל
model = FaceModel()
model.load_state_dict(torch.load("models_best_weights2.pt", weights_only=True))

model.eval()

# משנה את התמונה שתתאים למודל
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img_path = r'C:\Users\יאיר\PycharmProjects\pythonProject10\1000_F_170157165_Xp8pw5YnIPDhH3uBSyT87z8BS6yA69aT.jpg'
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
