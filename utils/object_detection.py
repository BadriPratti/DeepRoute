from ultralytics import YOLO
import cv2

device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
yolo_model = YOLO("/home/oshkosh/Depth-Anything-V2/yolov8n.pt").to(device)

def detect_objects(frame):
    results = yolo_model.predict(frame, device=device)
    return results[0].boxes

def draw_detection(frame, x1, y1, x2, y2, confidence, depth_value):
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    label = f"{confidence:.2f} | Depth: {depth_value:.2f}m"
    cv2.putText(frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
