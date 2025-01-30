
RoboflowASLDataset
------------------

This dataset was exported from Roboflow: (https://universe.roboflow.com/asl-2/asl-new/dataset/3)

The dataset includes 24173 images, divided into:
Train: 22707
Valid: 961
Test: 505

The downloaded images were annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Resize to 640x640 (Stretch)
* Grayscale (CRT phosphor)

The following augmentations were applied to the bounding boxes of each image:
* Random Gaussian blur of between 0 and 2 pixels
* Salt and pepper noise was applied to 0.1 percent of pixels


RoboflowASLAphabetDataset
-------------------------

This is the dataset we used for fine tuning YOLO models. We preprocessed the RoboflowASLDataset by dropping the word classes and keeping the 26 alphabet classes. The pre-processed dataset, RoboflowASLAphabetDataset, consists of 2988 images, divided into:

Train: 2606
Valid: 265
Test: 117

All the images were annotated in YOLOv8 format.


