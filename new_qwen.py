import cv2
from ultralytics import YOLO
import numpy as np
from datetime import datetime
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
import sys

# Prompt user for the object to detect
target_object = input("Enter the object you want to detect (e.g., person, car, dog): ").strip().lower()

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Pre-trained YOLOv8 nano model

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 is typically the default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit(1)

# Object detection loop
image_path = None
try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Perform object detection
        results = model(frame)

        # Process results
        target_detected = False
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
            confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
            class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs
            names = result.names  # Class names

            # Draw bounding boxes
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                if conf > 0.5:  # Confidence threshold
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{names[cls_id]} {conf:.2f}"
                    if names[cls_id].lower() == target_object:
                        # Draw red rectangle for target object
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        target_detected = True
                    else:
                        # Draw green rectangle for other objects
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Put label above the rectangle
                    cv2.putText(frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if names[cls_id].lower() == target_object else (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Object Detection', frame)

        # Save image and exit if target object is detected
        if target_detected:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = f"detected_{target_object}_{timestamp}.jpg"
            cv2.imwrite(image_path, frame)
            print(f"Image saved as {image_path}")
            break

        # Check for 'q' key or window close event
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or cv2.getWindowProperty('Object Detection', cv2.WND_PROP_VISIBLE) < 1:
            break

finally:
    # Release webcam resources
    cap.release()
    cv2.destroyAllWindows()

# Start chatbot if an image was saved
if image_path:
    print("Starting chatbot. Ask questions about the image or anything else. Type 'quit' to exit.")
    
    # Check if GPU is available, else fallback to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        # Load Qwen-VL model and processor
        model_vl = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.float16 if device == "cuda" else torch.float32, device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

        # Load the saved image
        image = Image.open(image_path)

        # Generate a short description of the image
        description_prompt = "Provide a short description of the image."
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": description_prompt}
                ]
            }
        ]

        # Process the description prompt
        inputs = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(
            text=inputs,
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(device)

        # Generate description
        output_ids = model_vl.generate(**inputs, max_new_tokens=100)
        description = processor.decode(output_ids[0], skip_special_tokens=True)
        description = description.split("Assistant:")[-1].strip()
        print(f"\nImage Description: {description}\n")

        # Chatbot loop
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() == 'quit':
                print("Exiting chatbot.")
                break

            # Prepare the conversation input
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": user_input}
                    ]
                }
            ]

            # Process the input
            inputs = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(
                text=inputs,
                images=[image],
                padding=True,
                return_tensors="pt"
            ).to(device)

            # Generate response
            output_ids = model_vl.generate(**inputs, max_new_tokens=512)
            response = processor.decode(output_ids[0], skip_special_tokens=True)
            
            # Extract only the assistant's response
            response = response.split("Assistant:")[-1].strip()
            print(f"Chatbot: {response}")

    except Exception as e:
        print(f"Error loading or running Qwen-VL model: {e}")
        print("Chatbot could not be started. Ensure 'accelerate' is installed and system resources are sufficient.")

    finally:
        # Clean up model resources
        if 'model_vl' in locals():
            del model_vl
            if device == "cuda":
                torch.cuda.empty_cache()

else:
    print("No image was saved. Chatbot will not start.")
