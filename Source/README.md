
## Structure

fine-tuning-yolo.py: Script to fine-tune YOLO models on a specified dataset.

client-realtime-detection.py: Client-side code that captures hand images from a camera, preprocesses them, and then sends them to the server for prediction.

server-realtime-detection.py: Server-side code that receives images from the client, performs predictions using the YOLO model, and sends the results back to the client.

fine-tuned-yolo8.pt: The saved YOLOv8 model used for real-time detection and prediction.


## Requirements

Python 3.x
OpenCV
MediaPipe
Requests
Ultralytics YOLOv8
Flask (for the client-server setup)


## Install the required packages

bash

pip install opencv-python mediapipe ultralytics flask numpy requests


## Getting Started

1. Training the Model

To fine-tune the YOLO model on your dataset, use the fine-tuning-yolo.py script. 

python fine-tuning-yolo.py

This script will output a trained YOLO model saved in the runs directory.

Notes: 
- The model may be changed to YOLOv9m, YOLOv11m, or other YOLO versions.
- Update the dataset path "data_yaml" to the directory of the downloaded dataset.

2. Client-Server Setup

The client-server setup allows for remote processing, where the client captures and sends images to the server for prediction, and the server returns the prediction results.

## Running the Server

To start the server, run server-realtime-detection.py. 

python server-realtime-detection.py

The server will start listening for incoming requests on the IP address and port.

Note: The model_path should be updated to the ".pt" file path


## Running the Client

The client-realtime-detection.py script captures images from a connected camera, preprocesses them, and sends them to the server for prediction.

Notes:
- The "response = requests.post" should include the server IP address
- The script is configured for a single camera source. Adjust the camera index in cv2.VideoCapture() if necessary.
