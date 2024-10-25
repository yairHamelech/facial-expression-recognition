import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from torch.utils.data import DataLoader
import timm
from torch import nn
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import confusion_matrix
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


Train_Image_folder_path = './images/train/'
Valid_Image_folder_path = './images/validation/'


Lr = 0.00001
Batch_size = 64
epochs = 15  # האפוקס להרצה הראשונה  (לא חשובים באמת כרגע)

additional_epochs = 1  # אפוקס נוספים לכל הרצה נוספת
model_name = "efficientnetb0"

# שינו מאגר הנתונים ליצור יותר אפדרויות בשביל האמון
train_augs = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=(-20, 20)),

    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

validation_augs = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# מאגר נתונים
trainset = ImageFolder(Train_Image_folder_path, transform=train_augs)
validset = ImageFolder(Valid_Image_folder_path, transform=validation_augs)
trainloader = DataLoader(trainset, batch_size=Batch_size, shuffle=True)
validloader = DataLoader(validset, batch_size=Batch_size)


print("Class Distribution in Training Set:", Counter([label for _, label in trainset]))


class FaceModel(nn.Module):
    def __init__(self):
        super(FaceModel, self).__init__()
        self.eff_net = timm.create_model('efficientnet_b0', pretrained=True, num_classes=7)

    def forward(self, images, labels=None):
        logits = self.eff_net(images)
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return logits, loss
        return logits

# מוריד את המודל עם המשקלים מהאימון הקודם כדי שיהיה כבר משקלים בהתחלה ולא להתחיל בלי כלום
Model = FaceModel()
Model.load_state_dict(torch.load("models_best_weights2.pt", weights_only=True))  # מוריד את המשקלים הקודמים של המודל

Model.to(device)  # מעביר את המודל לgpu או cpu


optimizer = torch.optim.AdamW(Model.parameters(), lr=Lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

# פונקצייה של accuracy
def multiclass_accuracy(y_pred, y_true):
    top_p, top_class = y_pred.topk(1, dim=1)
    equals = top_class == y_true.view(*top_class.shape)
    return torch.mean(equals.type(torch.FloatTensor))

# פונקצייה שמראה כמה פעמים הוא צדק וכמה טעה לפי החיזוי ותשובה האמיתית לכל קלאס
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# פונקציית האימון
def train_fn(model, dataloader, optimizer, current_epo):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    tk = tqdm(dataloader, desc="epoch [train] " + str(current_epo + 1))

    for t, data in enumerate(tk):
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits, loss = model(images, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += multiclass_accuracy(logits, labels)
        tk.set_postfix({'loss': '%6f' % float(total_loss / (t + 1)), 'acc': '%6f' % float(total_acc / (t + 1))})

    return total_loss / len(dataloader), total_acc / len(dataloader)

# מראה את ההתקדמות בכל שלב בעזרת postfix לבסוף מריצה את plot_confusion_matrix
def eval_fn(model, dataloader, current_epo):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    all_preds = []
    all_labels = []
    tk = tqdm(dataloader, desc="epoch [valid] " + str(current_epo + 1))

    with torch.no_grad():
        for t, data in enumerate(tk):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            logits, loss = model(images, labels)
            total_loss += loss.item()
            total_acc += multiclass_accuracy(logits, labels)

            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            tk.set_postfix({'loss': '%6f' % float(total_loss / (t + 1)), 'acc': '%6f' % float(total_acc / (t + 1))})


    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    plot_confusion_matrix(all_labels, all_preds, classes)

    return total_loss / len(dataloader), total_acc / len(dataloader)

# לולאת אימון
best_valid_loss = np.inf
for i in range(epochs, epochs + additional_epochs):
    train_loss, train_acc = train_fn(Model, trainloader, optimizer, i)
    valid_loss, valid_acc = eval_fn(Model, validloader, i)

    scheduler.step(valid_loss)

    if valid_loss < best_valid_loss:
        torch.save(Model.state_dict(), "models_best_weights2.pt")
        print("Saved best model weights.")
        best_valid_loss = valid_loss


