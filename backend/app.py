import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io

# --- CONFIGURATION ---
CLASS_NAMES = [
    'Bell_Pepper_Bacterial_spot',
    'Bell_Pepper_Healthy',
    'Corn_Blight',
    'Corn_Common_Rust',
    'Corn_Healthy',
    'Gray_Leaf_Spot',
    'Potato_Early_Blight',
    'Potato_Healthy',
    'Potato_Late_Blight',
    'Rice_Bacterial_leaf_blight',
    'Rice_Brown_spot',
    'Rice_Leaf_smut',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Healthy',
    'Tomato_Late_blight'
]

# Initialize Flask
app = Flask(__name__)
CORS(app)  # Allows React to talk to this backend


# --- MODEL ARCHITECTURE (Must match Colab exactly!) ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        # Layer 1: 224 -> 112
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Layer 2: 112 -> 56
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # Layer 3: 56 -> 28
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Flatten: 64 * 28 * 28
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 64 * 28 * 28)  # Flatten
        x = F.relu(self.fc1(x))
        # Note: We don't need dropout for prediction/inference
        x = self.fc2(x)
        return x


# --- LOAD THE BRAIN ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=len(CLASS_NAMES))

# Load the weights
try:
    # map_location='cpu' ensures it loads even if you don't have a GPU locally
    model.load_state_dict(torch.load('agrolens_model_gpu.pth', map_location=torch.device('cpu')))
    model.to(device)
    model.eval()  # Set to evaluation mode (IMPORTANT!)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Double check that CLASS_NAMES matches the number of classes you trained on!")


# --- PREPROCESSING ---
def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return my_transforms(image).unsqueeze(0)  # Add batch dimension (1, 3, 224, 224)


# --- ROUTES ---
@app.route('/', methods=['GET'])
def home():
    return "AgroLens AI API is Running! 🚀"


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img_bytes = file.read()

    try:
        # 1. Preprocess
        tensor = transform_image(img_bytes).to(device)

        # 2. Predict
        outputs = model(tensor)
        _, preds = torch.max(outputs, 1)
        predicted_idx = preds.item()

        # 3. Confidence
        probs = F.softmax(outputs, dim=1)
        confidence = probs[0][predicted_idx].item() * 100

        return jsonify({
            'class_name': CLASS_NAMES[predicted_idx],
            'confidence': f"{confidence:.2f}%",
            'class_id': predicted_idx
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)