import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from torch.utils.data import DataLoader, WeightedRandomSampler
import timm
from torch import nn
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths to the dataset directories
Train_Image_folder_path = './data/train'
Valid_Image_folder_path = './data/val/'

# Hyperparameters
Lr = 0.001  # Updated learning rate for better convergence with Adam
Batch_size = 32
epochs = 5
additional_epochs = 10
model_name = "efficientnet_b7"  # Changed to ResNet50

# Data augmentations for training and validation
train_augs = T.Compose([

    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(20),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


validation_augs = T.Compose([

    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
trainset = ImageFolder(Train_Image_folder_path, transform=train_augs)
validset = ImageFolder(Valid_Image_folder_path, transform=validation_augs)
print(trainset.class_to_idx)
# Calculate class counts to create class weights for imbalance
# Class counts based on trainset.class_to_idx
class_counts = {
    'angry': 10108,
    'disgust': 2913,
    'fear': 7279,
    'happy': 7244,
    'neutral': 7211,
    'sad': 7163,
    'surprise': 8029
}

# Map counts based on the actual class indices assigned by ImageFolder
class_weights = [1.0 / class_counts[class_name] for class_name in trainset.class_to_idx.keys()]
weights = [class_weights[label] for label in trainset.targets]  # Assign weights to each sample
sampler = WeightedRandomSampler(weights, num_samples=len(trainset), replacement=True)


# Create DataLoader using the sampler
trainloader = DataLoader(trainset, batch_size=Batch_size, sampler=sampler)
validloader = DataLoader(validset, batch_size=Batch_size)

# Define the model with ResNet-50
class FaceModel(nn.Module):
    def __init__(self):
        super(FaceModel, self).__init__()
        # Use efficientnet_b7 as base model
        self.eff_net = timm.create_model('tf_efficientnet_b7.aa_in1k', pretrained=True, num_classes=7)

        # Modify the final layer (classifier) to use more neurons
        self.eff_net.classifier = nn.Sequential(
            nn.Linear(self.eff_net.classifier.in_features, 2304),  # Increased to 4096 neurons
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2304, 7)  # Final output for 7 classes
        )

    def forward(self, images, labels=None):
        logits = self.eff_net(images)
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return logits, loss
        return logits



# Load previous model weights if available
Model = FaceModel()
Model.to(device)

# Optimizer and scheduler
optimizer = torch.optim.AdamW(Model.parameters(), lr=Lr, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=3e-4,
    total_steps=len(trainloader) * epochs,
    pct_start=0.3
)


# Confusion matrix plotting


# Accuracy function
def multiclass_accuracy(y_pred, y_true):
    top_p, top_class = y_pred.topk(1, dim=1)
    equals = top_class == y_true.view(*top_class.shape)
    return equals.sum().item()

# Training function
def train_fn(model, dataloader, optimizer, current_epoch):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    tk = tqdm(dataloader, desc=f"Epoch [train] {current_epoch + 1}")

    for data in tk:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits, loss = model(images, labels)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

        tk.set_postfix({'loss': f'{total_loss/(total_samples/Batch_size):.4f}', 'acc': f'{(total_correct/total_samples):.4f}'})

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

# Evaluation function
def eval_fn(model, dataloader, current_epoch):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    tk = tqdm(dataloader, desc=f"Epoch [valid] {current_epoch + 1}")

    with torch.no_grad():
        for data in tk:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            logits, loss = model(images, labels)
            total_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            tk.set_postfix({'loss': f'{total_loss/(total_samples/Batch_size):.4f}', 'acc': f'{(total_correct/total_samples):.4f}'})

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)



    print(f"F1-Score: {f1_score(all_labels, all_preds, average='weighted'):.4f}")

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

# Initialize best valid loss to track improvements
best_valid_loss = np.inf
model_weights_path = "models_best_weights.pt"
start_epoch = 0

if os.path.exists(model_weights_path):
    print(f"Loading model weights from {model_weights_path}...")
    Model.load_state_dict(torch.load(model_weights_path))
    start_epoch = epochs

# Training loop for the initial epochs
for epoch in range(start_epoch, epochs):
    print(f"Training epoch {epoch+1}/{epochs}...")
    train_loss, train_acc = train_fn(Model, trainloader, optimizer, epoch)
    valid_loss, valid_acc = eval_fn(Model, validloader, epoch)

    scheduler.step()

    if valid_loss < best_valid_loss:
        torch.save(Model.state_dict(), model_weights_path)
        print("Saved best model weights.")
        best_valid_loss = valid_loss

# After initial epochs, train for additional epochs if specified
if additional_epochs > 0:
    if os.path.exists(model_weights_path):
        print(f"Reloading best model weights from {model_weights_path} before additional epochs...")
        Model.load_state_dict(torch.load(model_weights_path))

    print(f"Training for {additional_epochs} additional epochs...")
    for epoch in range(epochs, epochs + additional_epochs):
        print(f"Training additional epoch {epoch+1}/{epochs + additional_epochs}...")
        train_loss, train_acc = train_fn(Model, trainloader, optimizer, epoch)
        valid_loss, valid_acc = eval_fn(Model, validloader, epoch)

        scheduler.step()

        if valid_loss < best_valid_loss:
            torch.save(Model.state_dict(), model_weights_path)
            print("Saved best model weights.")
            best_valid_loss = valid_loss
