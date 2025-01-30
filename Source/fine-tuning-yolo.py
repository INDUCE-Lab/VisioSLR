import ultralytics
import torch

# Check if there are at least 2 GPUs available
if torch.cuda.device_count() >= 2:
    # Set up device IDs for multi-GPU training
    devices = '0,1'  # You can specify more GPUs if available

    # Load the YOLOv8 model
    model = ultralytics.YOLO('yolov8m.pt')  # Load a YOLO pre-trained model (YOLOv8m, YOLOv9m, YOLOv11m)

    # Define the path to your dataset's YAML file
    data_yaml = r'/path/to/ASL new.v3i.yolov8/data.yaml' 

    # Train the model with multi-GPU support
    model.train(data=data_yaml, epochs=50, imgsz=640, batch=16, device=devices)

    # Save the trained model
    # If you fine-tune with YOLOv9 or YOLOv11, change 8 to 9 or 11
    model.save('fine-tuned-yolo8.pt') 
else:
    print("Insufficient GPUs available. At least 2 GPUs are required for parallel training.")
