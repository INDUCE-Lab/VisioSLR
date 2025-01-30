import cv2  # Import the OpenCV library for image and video processing.
import mediapipe as mp  # Import MediaPipe, a library for computer vision tasks like hand detection.
import numpy as np  # Import NumPy for array manipulation.
import requests  # Import the requests library to make HTTP requests.
import time  # Import the time library for delay functionalities.

# Initialize the MediaPipe Hands solution
mp_hands = mp.solutions.hands  # Access the Hands module from MediaPipe.
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
# Create a hand detector with real-time processing, limited to one hand, and minimum detection confidence of 0.5.

# Start video capture from the camera
cap = cv2.VideoCapture(0)  # Start capturing video from the default camera (ID 0).

# Set delay time (in seconds)
delay_time = 0.03  # Set a small delay time for processing each frame.

try:
    while True:  # Infinite loop for continuous frame processing.
        ret, frame = cap.read()  # Read a frame from the camera.
        if not ret:  # Check if frame capture failed.
            print("Failed to capture image")  # Print an error message if no frame is captured.
            break  # Exit the loop if frame capture fails.

        # Convert the frame to RGB format required by MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame from BGR to RGB.

        # Process the frame to detect hands
        results = hands.process(rgb_frame)  # Detect hands and landmarks in the RGB frame.

        # If hand landmarks are detected
        if results.multi_hand_landmarks:  # Check if hands were detected.
            for hand_landmarks in results.multi_hand_landmarks:  # Loop over each detected hand.
                # Get bounding box coordinates for the hand
                image_height, image_width, _ = frame.shape  # Get frame dimensions.
                x_min = min([lm.x for lm in hand_landmarks.landmark]) * image_width  # Minimum x coordinate.
                x_max = max([lm.x for lm in hand_landmarks.landmark]) * image_width  # Maximum x coordinate.
                y_min = min([lm.y for lm in hand_landmarks.landmark]) * image_height  # Minimum y coordinate.
                y_max = max([lm.y for lm in hand_landmarks.landmark]) * image_height  # Maximum y coordinate.

                # Expand the bounding box by a margin (e.g., 20% of the box size)
                margin = 0.2  # Set margin as 20% of the box size.
                box_width = x_max - x_min  # Calculate bounding box width.
                box_height = y_max - y_min  # Calculate bounding box height.
                x_min = max(0, int(x_min - margin * box_width))  # Adjust x_min to include margin.
                x_max = min(image_width, int(x_max + margin * box_width))  # Adjust x_max with margin.
                y_min = max(0, int(y_min - margin * box_height))  # Adjust y_min with margin.
                y_max = min(image_height, int(y_max + margin * box_height))  # Adjust y_max with margin.
                # Draw the bounding box around the detected hand
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Draw the bounding box.

                # Crop the hand region with the expanded bounding box
                cropped_hand = frame[y_min:y_max, x_min:x_max]  # Crop the hand region from the frame.

                # Resize and pad the image to a fixed size for model input
                target_size = 640  # Define target size for the image based on the trained model.
                height, width = cropped_hand.shape[:2]  # Get cropped image dimensions.
                scale = target_size / max(height, width)  # Calculate scaling factor.
                new_w, new_h = int(width * scale), int(height * scale)  # Calculate new dimensions.
                resized_hand = cv2.resize(cropped_hand, (new_w, new_h))  # Resize the cropped hand.
                padded_hand = np.full((target_size, target_size, 3), 128, dtype=np.uint8)  # Create padding.
                y_offset, x_offset = (target_size - new_h) // 2, (target_size - new_w) // 2  # Calculate offsets.
                padded_hand[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_hand  # Place resized hand.

                # Encode the image as JPEG
                _, img_encoded = cv2.imencode('.jpg', padded_hand)  # Encode the image as a JPEG byte stream.

                # Send the image to the server for prediction
                response = requests.post('http://XXX.XX.XX.XXX:XXXX/predict', data=img_encoded.tobytes()) # Update the IP address
                # Send the encoded image to the server for hand gesture prediction.

                # Process the response from the server
                if response.status_code == 200:  # Check if server response is successful.
                    result = response.json()  # Parse the JSON response.
                    prediction_label = result['label']  # Extract prediction label from response.
                    confidence = result['confidence']  # Extract confidence score from response.
                else:
                    print("Failed to get prediction from server")  # Error message if server request fails.
                    prediction_label, confidence = "N/A", 0.0  # Default values if request fails.

                # Draw the prediction on the frame
                cv2.rectangle(frame, (10, 30), (300, 80), (0, 0, 0), -1)  # Draw a rectangle for text background.
                cv2.putText(frame, f"{prediction_label}: {confidence:.2f}",
                            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)  # Display prediction.

        # Display the frame with annotations
        cv2.imshow("Client - Hand Detection", frame)  # Show the processed frame in a window.

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Break the loop if 'q' key is pressed.
            break  # Exit the loop.

finally:
    cap.release()  # Release the video capture resource.
    cv2.destroyAllWindows()  # Close all OpenCV windows.
    hands.close()  # Close the MediaPipe Hands solution.
