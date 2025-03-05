from flask import Flask, request, jsonify
import torch
import joblib
import numpy as np
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS so the extension can call the API

class RatingPredictor(torch.nn.Module):
    def __init__(self):
        super(RatingPredictor, self).__init__()
        self.fc1 = torch.nn.Linear(4, 16)
        self.fc2 = torch.nn.Linear(16, 8)
        self.fc3 = torch.nn.Linear(8, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load trained model and scalers
model = RatingPredictor()
model.load_state_dict(torch.load("rating_predictor.pth", map_location=torch.device('cpu')))
model.eval()

scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Incoming Request:", request.data)  # Debugging

        if not request.is_json:
            return jsonify({"error": "Request must be in JSON format"}), 400

        data = request.get_json()

        # Debugging: Print parsed JSON
        # print("Parsed JSON Data (Before Cleaning):", data)

        # Ensure all fields exist and remove any non-breaking spaces
        required_keys = ["time", "people_solved_A", "solved_at_time", "question_number"]
        for key in required_keys:
            if key not in data:
                return jsonify({"error": f"Missing required field: {key}"}), 400
            if isinstance(data[key], str):  # If a field is a string, clean it
                data[key] = data[key].replace("\xa0", "").strip()

        # Convert to integer where necessary
        data["time"] = int(data["time"])
        data["people_solved_A"] = int(data["people_solved_A"])
        data["solved_at_time"] = int(data["solved_at_time"])
        data["question_number"] = int(data["question_number"])

        # Debugging: Print cleaned JSON
        # print("Parsed JSON Data (After Cleaning):", data)

        # Convert input to NumPy array
        input_data = np.array([[data["time"], data["people_solved_A"], data["solved_at_time"], data["question_number"]]])

        # Scale input
        input_scaled = scaler_X.transform(input_data)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

        with torch.no_grad():
            predicted_scaled = model(input_tensor).numpy()

        predicted_rating = float(scaler_y.inverse_transform(predicted_scaled.reshape(-1, 1))[0][0])

        return jsonify({"predicted_rating": predicted_rating})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
