# real-time-object-detector-with-VLM

## install libraries and dependencies

### Install the OpenCV library for image and video processing.
Command: `pip install opencv-python opencv-python-headless`

### Install the ultralytics package to use the YOLO model for object detection.
Command: `pip install ultralytics`

### Install the transformers library for the BLIP model (used for image captioning or visual question answering).
Command: `pip install transformers`

### Install PyTorch, which is required for both ultralytics and transformers.
Command: `pip install torch torchvision torchaudio`

### Install NumPy for numerical operations and array handling.
Command: `pip install numpy`

### features
Capture: Handled by OpenCV

Detect: YOLOv8 performs detection with bounding boxes

Classify: detected objects by generating descriptions using the BLIP model.
