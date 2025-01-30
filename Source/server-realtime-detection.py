from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

# Initialize the YOLOv8 model
model_path = "path/to/fine-tuned-yolo.pt"
model = YOLO(model_path)

# Initialize an app ("Flask") to handle web requests; `__name__` helps locate app resources.
app = Flask(__name__)

# `@app.route` defines a URL endpoint for the app; here, '/predict' handles requests to this path.
# `methods=['POST']` specifies that this route only accepts POST requests, typically used for sending data.
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Decode the received image
        image_data = np.frombuffer(request.data, np.uint8)  # Convert binary image data from the request to a NumPy array with 8-bit unsigned integers
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)  # Decode the NumPy array as an image in color format using OpenCV

        # YOLO model prediction
        results = model.predict(image)[0]

        # Check if there's a valid prediction
        if results.boxes and results.names: # The trained model has the boxes and names saved
            class_id = int(results.boxes.cls[0]) # Get the classification ID ( the alphabet ID ) based on the trained model
            prediction_label = results.names[class_id]  # Show the assigned alphabet associated with the predicted classification (example: 1 = letter A)
            confidence = results.boxes.conf[0]  # Get the confidence score for the first detected object in `results`
            # This score indicates the model's certainty about the detection, based on the model's internal probability output
            # It’s not a direct measure of accuracy or precision but reflects the likelihood that the detection is correct
        else: # If we cant predict any alphabet, then return "unknown" with 0 confidence
            prediction_label = "Unknown"
            confidence = 0.0

        return jsonify({"label": prediction_label, "confidence": float(confidence)})
    # For debugging
    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Start the Flask app, making it accessible externally on port 5000.
    # `host='0.0.0.0'` allows connections from any IP address, making the app accessible to other devices on the network.
    # `port=5000` specifies the port where the app listens for incoming requests.

