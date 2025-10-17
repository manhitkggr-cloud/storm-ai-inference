from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io, base64

app = Flask(__name__)

# === Mô hình Encoder ===
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 128)
        )

    def forward(self, x):
        return self.encoder(x)

# === Load model ===
device = torch.device("cpu")
model = Encoder().to(device)
model.load_state_dict(torch.load("model/deep_svdd_bottle3.pth", map_location=device))
model.eval()
center_c = torch.load("model/center_c.pt", map_location=device)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# === Endpoint predict ===
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        img_b64 = data.get("image")
        if not img_b64:
            return jsonify({"error": "Missing image field"}), 400

        # Decode ảnh từ base64
        img_data = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            z = model(img_tensor)
            score = torch.sum((z - center_c) ** 2).item()

        label = 1 if score > 5.3e-06 else 0
        return jsonify({
            "label": label,
            "score": score,
            "message": "Anomaly" if label == 1 else "Normal"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "✅ AI Inference API running"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)