# naruto-hand-signs

Recognizes Naruto hand signs from a live webcam feed. Nothing fancy, just MediaPipe doing the hand tracking and a small MLP doing the classification on top.

![demo](readme-assets/demo.gif)

The idea is pretty simple: MediaPipe's HandLandmarker pulls out 21 hand landmarks per hand, those get normalized (wrist-relative, scale-normalized) and concatenated by handedness (left/right), and that vector gets fed into a small PyTorch MLP (128 → 64 → num_classes, with dropout) that spits out a sign class. All the landmark extraction happens once and gets cached into a `features.npz` so training doesn't have to redo it every run.

![hand signs](readme-assets/image.png)

Dataset-wise, images where MediaPipe can't detect any hands get dropped since there's nothing to extract, which throws away a decent chunk of the raw images. Worth keeping in mind if you're building on top of this or extending the dataset.

If you want the full walkthrough with the reasoning, plots, and failure cases, I wrote it up here: https://medium.com/@roha-pathan125/naruto-hand-signs-recognition-using-cnns-yolo-mediapipe-mlp-a71d6b3ab4b4?sharedUserId=roha-pathan125

### what's in here

- `live.py` — runs inference in real time off your webcam
- `train.ipynb` — the actual training pipeline, extraction → training → eval, all in one place
- `take-images.py` — helper script to build your own dataset from scratch
- `model.pt` — the trained model