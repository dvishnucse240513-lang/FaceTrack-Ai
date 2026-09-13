import face_recognition
import numpy as np
import cv2

FACE_DETECT_SCALE = 0.5
MIN_FACE_SIZE = (50, 50)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face_locations(image):
    """Detect faces quickly and return full-size RGB face locations."""
    small = cv2.resize(image, (0, 0), fx=FACE_DETECT_SCALE, fy=FACE_DETECT_SCALE)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=MIN_FACE_SIZE
    )
    if len(faces):
        scale = int(1 / FACE_DETECT_SCALE)
        return [(y * scale, (x + w) * scale, (y + h) * scale, x * scale) for (x, y, w, h) in faces]

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return face_recognition.face_locations(rgb_image, model='hog')

def get_face_embedding(image):
    """
    Extract face embedding from an image (numpy array in BGR format).
    Returns the first face embedding if found, else None.
    """
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = detect_face_locations(image)
    if not face_locations:
        return None
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    if face_encodings:
        return face_encodings[0]  # Return the first face embedding
    return None

def compare_embeddings(emb1, emb2, threshold=0.6):
    """
    Compare two face embeddings using Euclidean distance.
    Returns True if they match (distance < threshold), else False.
    """
    distance = np.linalg.norm(emb1 - emb2)
    return distance < threshold
