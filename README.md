# Real-time-object-detector-with-VLM
Here is how this project works. It opens the camera, searches for the required object (detected objects is in green boxing and required object is in red boxing), takes a picture of the screen once the required object is found and stops the camera. The VLM will provide a short description of the image and open a chatbot for the user to ask questions related to the required object and the image.

Capture: Handled by OpenCV

Detect & Classify: YOLOv8 performs detection with bounding boxes (added Define valid colors for filtering)

chatbot: Qwen-VL (it will only answer questions about the image & if there is some words mispelled the chatbot will guess what the user is trying to ask)

## functions
Pro: high speed in movement, color detected, high accuracy
Cons: can only ask for specific object of the different colors (e.g. red cup, black cup)

## Install libraries and dependencies 

### Install the OpenCV library for image and video processing.
Command: `pip install opencv-python opencv-python-headless`

### Install the ultralytics package to use the YOLO model for object detection.
Command: `pip install ultralytics`

### Install the transformers library for the BLIP model (used for image captioning or visual question answering).
Command: `pip install transformers`

### Install PyTorch, which is required for both ultralytics and transformers.
#### For faster setup (preferred)
`pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 torchaudio==2.2.1+cu121 --index-url https://download.pytorch.org/whl/cu121`

#### For normal setup
Command: `pip install torch torchvision torchaudio`

### Install NumPy for numerical operations and array handling.
Command: `pip install numpy`

### install win32gui and win32con
Command: `pip install pywin32`

## run the code
### create a virtual environmenr
`python -m venv venv`
`.\venv\Scripts\activate`
