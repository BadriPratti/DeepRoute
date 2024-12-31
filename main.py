import cv2
from utils.camera import is_camera_available, get_frame
from utils.object_detection import detect_objects, draw_detection
from utils.depth_processing import load_depth_model, estimate_depth
from llm.llm_engine import query_llm
from utils.speech import speak
from ultralytics import YOLO  # Import YOLO to access the model's class names

# Load YOLO model to access class names for mapping
yolo_model = YOLO('/home/oshkosh/Depth-Anything-V2/yolov8n.pt')  # Update with the correct path to your model

def main_loop():
    # Check if the camera is available
    if not is_camera_available():
        print("No camera detected. Please connect a camera.")
        return

    print("Loading Depth-Anything V2 model...")
    depth_model = load_depth_model()

    # Try different camera indices if the default (0) fails
    print("Starting webcam feed...")
    cap = None
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera detected at index {i}")
            break

    if not cap or not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not capture frame.")
            break
        
        # Resize frame to maintain consistent input size
        resized_frame = cv2.resize(frame, (640, 480))
        
        # Run object detection on the frame
        detections = detect_objects(resized_frame)
        
        # Estimate depth for detected objects
        depth_map = estimate_depth(depth_model, resized_frame)
        
        detected_objects = []
        
        # Process detected objects to extract depth and bounding box info
        for detection in detections:
            x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()
            confidence = detection.conf.cpu().item()

            # Calculate center of the bounding box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Map class ID to class name using YOLO model
            class_id = int(detection.cls[0].cpu().numpy())
            class_name = yolo_model.names.get(class_id, "unknown")  # Get class name

            # Get depth at the object's center
            if 0 <= center_x < depth_map.shape[1] and 0 <= center_y < depth_map.shape[0]:
                depth_value = depth_map[center_y, center_x]
            else:
                depth_value = 0.0

            # Append object information for prompt building
            detected_objects.append({
                "object": class_name,  # Use class name instead of detection.name
                "distance": f"{depth_value:.2f} meters",
                "position": "left" if center_x < frame.shape[1] // 2 else "right"
            })

            # Draw bounding box and annotate depth information
            draw_detection(resized_frame, x1, y1, x2, y2, confidence, depth_value)

        # Prepare and send context to the LLM for feedback
        if detected_objects:
            print("Objects detected, querying LLM...")
            prompt = build_prompt(detected_objects)
            response = query_llm(prompt)
            print(f"GPT Response: {response}")

            # Speak the generated response using TTS
            speak(response)

        # Display video feed with detections
        cv2.imshow("Object Detection & Depth Estimation", resized_frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close display window
    cap.release()
    cv2.destroyAllWindows()

# Build structured LLM prompt based on detected objects
def build_prompt(detected_objects):
    object_descriptions = [
        f"{obj['object']} detected at {obj['distance']} on your {obj['position']}"
        for obj in detected_objects
    ]
    
    prompt = f"""
    The following objects are detected:
    {', '.join(object_descriptions)}.
    Provide navigation feedback and warn about obstacles.
    """
    return prompt

if __name__ == "__main__":
    main_loop()
