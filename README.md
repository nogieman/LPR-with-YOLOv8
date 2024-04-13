

# Vehicle and License Plate Detection with YOLOv8

This repo includes the main project and the training module for the YOLOv8 model with custom dataset. The training file has to be edited to fit the proper location of data.yaml file and the dataset location.
The project combines YOLOv8 for vehicle and license plate detection with the SORT (Simple Online and real-time tracking) algorithm for vehicle tracking. It detects vehicles in a video stream using YOLOv8, tracks them across frames with SORT, and identifies license plates using another YOLOv5 model trained specifically for license plate detection. The project is implemented in Python using PyTorch and OpenCV.

## Dependencies

- [Ultralytics YOLO](https://github.com/ultralytics/yolov8)
- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [SORT (Simple Online and Realtime Tracking)](https://github.com/abewley/sort)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [OpenCV](https://opencv.org/)

## Overview

The project performs the following tasks:

1. **Vehicle Detection**: Uses a pre-trained YOLOv8 model to detect vehicles in a video stream. It draws bounding boxes around the detected vehicles.

2. **Vehicle Tracking**: Utilizes the SORT algorithm to track the detected vehicles across frames. It assigns unique IDs to each detected vehicle for tracking.

3. **License Plate Detection**: Employs a custom-trained YOLOv8 model specifically for license plate detection to identify license plates on the detected vehicles. It draws bounding boxes around the license plates and displays the recognized license plate numbers.

4. **License Plate Recognition**: Utilizes EasyOCR to perform Optical Character Recognition (OCR) on the detected license plates and extract the alphanumeric characters. The OCR_Use file is specified with the pattern of the Indian License plate.
   
5. **Output Video**: Writes the processed frames with bounding boxes and license plate numbers to an output video file.

## Usage

1. **Setup**: Install the required dependencies mentioned above.

2. **Download Pre-trained Models**: Download the pre-trained YOLOv8 models for vehicle detection and license plate detection.

3. **Run the Code**: Execute the provided Python script in a suitable environment, ensuring the availability of GPU for optimal performance.

4. **Input Video**: Provide the path to the input video file for vehicle and license plate detection.

5. **Output Video**: Specify the name and location for the output video file.

6. **Visualization**: Visualize the vehicle detection, tracking, and license plate recognition results in the output video.

## Results

The project achieves accurate vehicle detection, tracking, and license plate recognition in real-time video streams. It provides a valuable tool for various applications, including traffic monitoring, parking management, and law enforcement.

