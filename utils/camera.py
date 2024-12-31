import cv2

def is_camera_available(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return False
    cap.release()
    return True

def get_frame(camera_index=0, width=640, height=480):
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret:
            return cv2.resize(frame, (width, height))
    return None
