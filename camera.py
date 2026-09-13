import cv2

def get_frame():
    """
    Capture a single frame from the webcam.
    Returns the frame as numpy array (BGR) or None if failed.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None