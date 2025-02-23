from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T
import timm
import io
from flask_cors import CORS  # Import CORS

# Enable CORS for all routes
app = Flask(__name__)
CORS(app)  # This will allow all origins to access the server

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FaceModel(nn.Module):
    def __init__(self):
        super(FaceModel, self).__init__()
        # Use efficientnet_b7 as base model
        self.eff_net = timm.create_model('tf_efficientnet_b7.aa_in1k', pretrained=False, num_classes=7)

        # Modify the final layer (classifier) to use more neurons
        self.eff_net.classifier = nn.Sequential(
            nn.Linear(self.eff_net.classifier.in_features, 4096),  # Increased to 4096 neurons
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(4096, 7)  # Final output for 7 classes
        )

    def forward(self, images):
        return self.eff_net(images)

# Initialize model and load weights
model = FaceModel()
model.load_state_dict(torch.load("models_best_weights4.pt"))
model.to(device)
model.eval()  # Set to evaluation mode

# Define transformation for incoming images
transform = T.Compose([

    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint to handle image predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read the image file
        img = Image.open(io.BytesIO(file.read()))
        img = transform(img).unsqueeze(0).to(device)  # Transform and add batch dimension

        with torch.no_grad():
            outputs = model(img)
            softmax = torch.nn.Softmax(dim=1)
            probabilities = softmax(outputs)
            confidences = probabilities[0].cpu().numpy()  # Confidence scores

            _, predicted = torch.max(outputs, 1)
            predicted_class = predicted.item()

        # Class names for the facial expressions
        classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

        return jsonify({
            'prediction': classes[predicted_class],
            'confidence': confidences.tolist()  # Send confidence scores
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
