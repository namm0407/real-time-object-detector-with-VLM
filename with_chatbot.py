import cv2
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration, GPT2Tokenizer, GPT2LMHeadModel
import torch
import numpy as np
from datetime import datetime
import time

# Get user input for the target object
target_object = input("Enter the object you want to find: ").strip().lower()

# Initialize YOLOv8 for detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO("yolov8s.pt").to(device)  # Use pre-trained YOLOv8s model

# Initialize BLIP for classification/description
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Initialize GPT-2 for chatbot
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Ensure OpenCV window is created and focused
cv2.namedWindow("Real-Time Detection with BLIP")
cv2.setWindowProperty("Real-Time Detection with BLIP", cv2.WND_PROP_VISIBLE, 1)

frame_count = 0
photo_taken = False
image_description = ""

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Preprocess frame
        frame_resized = cv2.resize(frame, (640, 640))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Detect objects with YOLOv8
        results = model.predict(frame_rgb, conf=0.3, device=device)
        boxes = results[0].boxes.xyxy
        scores = results[0].boxes.conf
        class_names = results[0].names  # Pre-trained classes (e.g., 'cup', 'bottle')

        target_detected = False

        # Process detections and classify with BLIP
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = map(int, box)
            # Crop the detected region
            crop = frame_resized[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Prepare input for BLIP
            inputs = processor(images=crop, return_tensors="pt").to(device)
            outputs = blip_model.generate(**inputs, max_new_tokens=50)
            description = processor.decode(outputs[0], skip_special_tokens=True).lower()

            # Store description for chatbot if target is detected
            if target_object in description:
                image_description = description

            # Determine box color: red for target object, green for others
            is_target = target_object in description
            box_color = (0, 0, 255) if is_target else (0, 255, 0)  # Red for target, green for others

            # Visualize detection and BLIP description
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), box_color, 2)
            label = f"{description.split('.')[0]}: {score:.2f}"
            cv2.putText(frame_resized, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            # If target object is detected, set flag to capture photo
            if is_target:
                target_detected = True

        # Display frame
        cv2.imshow("Real-Time Detection with BLIP", frame_resized)

        # Check for key press
        key = cv2.waitKey(10) & 0xFF
        print(f"Key pressed: {key}")  # Debug print
        if key == ord("q") or key == ord("Q"):  # Exit on 'q' or 'Q'
            print("Exiting on 'q' key")
            break
        elif key in [13, 32]:  # Capture photo on 'Enter' or 'Space' and exit
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"manual_capture_{timestamp}.jpg"
            cv2.imwrite(filename, frame_resized)
            print(f"Image saved as {filename}")
            break

        # Capture photo if target object is detected
        if target_detected and not photo_taken:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detected_{target_object}_at_{timestamp}.jpg"
            cv2.imwrite(filename, frame_resized)
            print(f"Image with {target_object} saved as {filename}")
            photo_taken = True
            break

        frame_count += 1

except KeyboardInterrupt:
    print("Program terminated by user (Ctrl+C or similar).")

# Start chatbot if an image was captured with the target object
if photo_taken and image_description:
    print(f"\nChatbot activated! The image contains: {image_description}")
    print("Ask questions about the image (type 'exit' to quit):")
    
    while True:
        user_question = input("You: ").strip()
        if user_question.lower() == "exit":
            print("Exiting chatbot.")
            break

        # Prepare prompt for GPT-2
        prompt = f"The image shows: {image_description}. Question: {user_question} Answer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = gpt2_model.generate(
            inputs["input_ids"],
            max_length=150,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the answer part (remove the prompt)
        answer = response[len(prompt):].strip()
        print(f"Chatbot: {answer}")

else:
    print("No target object detected or no image captured. Chatbot not started.")

# Cleanup camera and OpenCV windows
cap.release()
cv2.destroyAllWindows()
