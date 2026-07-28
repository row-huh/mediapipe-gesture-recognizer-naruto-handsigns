# mediapipe landmarks = mlp = win?

# Capture & Detect: Use OpenCV to read video frames and pass them to
# MediaPipe Hands to extract 21 3D landmark points (x, y, z) per frame.


# Normalize Coordinates: Convert absolute pixel values into relative coordinates by 
# subtracting the wrist landmark position (x₀, y₀, z₀) and scaling the vector by 
# the maximum distance from the wrist to make the gesture invariant to hand 
# size and distance from the camera.


"""
gesture_pipeline.py  (HandLandmarker / Tasks API version)
Usage:
  python gesture_pipeline.py extract --data_dir ./dataset --out features.npz
  python gesture_pipeline.py train   --features features.npz --out model.pt
  python gesture_pipeline.py infer   --model model.pt --image test.jpg
"""

import os, argparse, urllib.request
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import cv2
import mediapipe as mp



MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand_landmarker.task ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def make_landmarker(mode="IMAGE"):
    base_options = mp_python.BaseOptions(model_asset_path=ensure_model())
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE if mode == "IMAGE" else vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def landmarks_to_vec(hand_landmarks):
    coords = np.array([[p.x, p.y, p.z] for p in hand_landmarks], dtype=np.float32)  # (21,3)
    coords -= coords[0]  # wrist-relative
    scale = np.max(np.linalg.norm(coords, axis=1))
    if scale > 1e-6:
        coords /= scale
    return coords.flatten()  # 63-d


# ---------- EXTRACTION ----------
def extract_landmarks(data_dir, out_path):

    landmarker = make_landmarker("IMAGE")

    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_to_idx = {c: i for i, c in enumerate(classes)}

    X, y = [], []
    skipped = 0
    for c in classes:
        folder = os.path.join(data_dir, c)
        for fname in os.listdir(folder):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(folder, fname)
            img = cv2.imread(path)
            if img is None:
                skipped += 1
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            result = landmarker.detect(mp_image)

            if not result.hand_landmarks:
                skipped += 1
                continue

            X.append(landmarks_to_vec(result.hand_landmarks[0]))
            y.append(class_to_idx[c])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    print(f"Extracted {len(X)} samples, skipped {skipped} (no hand detected / unreadable)")
    np.savez(out_path, X=X, y=y, classes=np.array(classes))
    landmarker.close()


# ---------- MODEL ----------
def build_model(input_dim, num_classes):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, num_classes)
    )


# ---------- TRAIN ----------
def train(features_path, out_path, epochs=100, lr=1e-3, batch_size=32):
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader, random_split

    data = np.load(features_path, allow_pickle=True)
    X, y, classes = data["X"], data["y"], list(data["classes"])

    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    n_val = max(1, int(0.15 * len(dataset)))
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(X.shape[1], len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            torch.save({"state_dict": model.state_dict(),
                        "classes": classes,
                        "input_dim": X.shape[1]}, out_path)
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:3d} | loss {loss.item():.4f} | val_acc {acc:.4f}")

    print(f"Best val_acc: {best_acc:.4f} | saved to {out_path}")


# ---------- INFER ----------
def infer(model_path, image_path):
    import cv2, torch
    import mediapipe as mp

    ckpt = torch.load(model_path, map_location="cpu")
    classes = ckpt["classes"]
    model = build_model(ckpt["input_dim"], len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    landmarker = make_landmarker("IMAGE")
    img_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        print("No hand detected")
        return

    x = torch.tensor(landmarks_to_vec(result.hand_landmarks[0])).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
        idx = probs.argmax().item()
    print(f"Prediction: {classes[idx]} ({probs[idx].item():.3f})")
    landmarker.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract")
    pe.add_argument("--data_dir", required=True)
    pe.add_argument("--out", default="features.npz")

    pt = sub.add_parser("train")
    pt.add_argument("--features", required=True)
    pt.add_argument("--out", default="model.pt")
    pt.add_argument("--epochs", type=int, default=100)

    pi = sub.add_parser("infer")
    pi.add_argument("--model", required=True)
    pi.add_argument("--image", required=True)

    args = p.parse_args()
    if args.cmd == "extract":
        extract_landmarks(args.data_dir, args.out)
    elif args.cmd == "train":
        train(args.features, args.out, epochs=args.epochs)
    elif args.cmd == "infer":
        infer(args.model, args.image)