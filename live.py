import os
import urllib.request
import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


MODEL_URL = "https://googleapis.com"
LANDMARKER_MODEL_PATH = "hand_landmarker.task"
PYTORCH_MODEL_PATH = "model.pt"
MAX_HANDS = 2

# --- 1. PYTORCH MODEL ARCHITECTURE RECREATION ---
def build_model(input_dim, num_classes):
    return nn.Sequential(
        nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, num_classes)
    )

# --- 2. MEDIAPEPIE DOWNLOAD & INITIALIZATION ---
def ensure_model():
    if not os.path.exists(LANDMARKER_MODEL_PATH):
        print("Downloading hand_landmarker.task ...")
        urllib.request.urlretrieve(MODEL_URL, LANDMARKER_MODEL_PATH)
    return LANDMARKER_MODEL_PATH

def make_landmarker():
    base_options = mp_python.BaseOptions(model_asset_path=ensure_model())
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE, # Sync processing per frame
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)

# --- 3. VECTOR PROCESSING PIPELINE ---
def single_hand_vec(hand_landmarks):
    coords = np.array([[p.x, p.y, p.z] for p in hand_landmarks], dtype=np.float32)
    coords -= coords[0]  # Wrist-relative centering
    scale = np.max(np.linalg.norm(coords, axis=1))
    if scale > 1e-6:
        coords /= scale
    return coords.flatten()  # 63-d

def landmarks_to_vec(result):
    if not result.hand_landmarks:
        return None
        
    if MAX_HANDS == 1:
        return single_hand_vec(result.hand_landmarks[0])

    slots = {"Left": np.zeros(63, dtype=np.float32), "Right": np.zeros(63, dtype=np.float32)}
    for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        label = handedness[0].category_name  # "Left" or "Right"
        slots[label] = single_hand_vec(hand_landmarks)
    return np.concatenate([slots["Left"], slots["Right"]])  # 126-d

# --- 4. OPENCV DRAWING UTILITY ---
def draw_skeleton(frame, hand_landmarks):
    # Setup standard color palettes for visual render
    h, w, _ = frame.shape
    connections = [
        (0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12), (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20), (5,9),(9,13),(13,17)
    ]
    # Draw bone links
    for start_idx, end_idx in connections:
        pt1 = (int(hand_landmarks[start_idx].x * w), int(hand_landmarks[start_idx].y * h))
        pt2 = (int(hand_landmarks[end_idx].x * w), int(hand_landmarks[end_idx].y * h))
        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
    # Draw joint points
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

# --- MAIN EXECUTION PIPELINE ---
def main():
    # Load your custom dictionary package
    print("Loading PyTorch classification checkpoints...")
    checkpoint = torch.load(PYTORCH_MODEL_PATH, map_location=torch.device("cpu"))
    
    input_dim = checkpoint["input_dim"]
    classes = checkpoint["classes"]
    num_classes = len(classes)
    
    # Rebuild, inject weights, and flip model state to evaluate mode
    model = build_model(input_dim, num_classes)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    # Fire up the MediaPipe pipeline
    print("Initializing MediaPipe Landmarker...")
    landmarker = make_landmarker()
    
    # Setup live video capture pipeline
    cap = cv2.VideoCapture(0)
    print("Live Feed Active. Press 'q' to exit the application loop.")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame pipeline glitch.")
            continue

        # Mirror output horizontally for intuitive tracking layout
        frame = cv2.flip(frame, 1)
        
        # Convert BGR camera stream feed to MediaPipe standard RGB Image object
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # Get hand landmarks
        result = landmarker.detect(mp_image)
        
        # Render visual bones if hands are present
        if result.hand_landmarks:
            for hand_lms in result.hand_landmarks:
                draw_skeleton(frame, hand_lms)
                
            # Process coordinates down into your target array shape
            features = landmarks_to_vec(result)
            
            if features is not None:
                # Wrap with batch array dim -> [1, dim_size]
                input_tensor = torch.tensor([features], dtype=torch.float32)
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                    
                    # 1. Strip away the batch dimension [0] to make it a clean 1D array
                    probabilities = torch.softmax(outputs, dim=1)[0] 
                    
                    # 2. Get the winning index from the 1D array
                    predicted_idx = torch.argmax(probabilities).item()
                    
                    # 3. Pull out the confidence score safely
                    confidence = probabilities[predicted_idx].item()

                
                # Fetch readable string identity mapped via classification dict
                gesture_name = classes[predicted_idx]
                
                # Draw predictions onto OpenCV frame UI layer
                display_text = f"Gesture: {gesture_name} ({confidence:.2%})"
                cv2.putText(frame, display_text, (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Display the frame Window
        cv2.imshow("Real-Time Gesture Classifier Pipeline", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
