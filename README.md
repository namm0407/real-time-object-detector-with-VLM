# Real-time-object-detector-with-VLM

## so far
switching to qwen model because the chatbot in with_chatbot.py have really low accuracy (it's not even answering the questions)

the code in qwen.py is working. But the movement is a bit slow and it is also color blinded.

## Files
### blip.py (Finished but no chatbot)
Capture: Handled by OpenCV

Detect: YOLOv8 performs detection with bounding boxes

Classify: detected objects by generating descriptions using the BLIP model (the wording of the boxes)

### with_chatbot.py (the code works but the repsonse of the chatbot is not accuracy)
Capture: Handled by OpenCV

Detect: YOLOv8 performs detection with bounding boxes

Classify: detected objects by generating descriptions using the BLIP model (the wording of the boxes)

Chatbot : Handled by Mixtral-8x7B

### chatbot2.py (trying other chatbots (In progess ...))

### qwen.py (In progess)
switched to qwen because the chatbot will have higher accuracy.

Capture: Handled by OpenCV

Detect: YOLOv8 performs detection with bounding boxes

Classify: detected objects by generating descriptions using the qwen-vl model 

### new.py (backup (without chatbot) )
Capture: Handled by OpenCV

Detect & Classify: YOLOv8 performs detection with bounding boxes

### new_qwen.py (finished)
Capture: Handled by OpenCV

Detect & Classify: YOLOv8 performs detection with bounding boxes

chatbot: Qwen-VL (it will only answer questions about the image & if there is some words mispelled the chatbot will guess what the user is trying to ask)

## Install libraries and dependencies (For the blip one)

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



