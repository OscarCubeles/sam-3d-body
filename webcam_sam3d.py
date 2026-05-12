import torch
print(torch.__version__)

import cv2
import numpy as np
import torch

from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator


# -----------------------
# Load model ONLY ONCE
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # USE fast_sam_3d_body environment

model, model_cfg = load_sam_3d_body(
    r"C:\Users\oscar\Documents\TU\TFM\sam-3d-body\checkpoints\sam-3d-body-dinov3\model.ckpt",
    device=device,
    mhr_path=r"C:\Users\oscar\Documents\TU\TFM\sam-3d-body\checkpoints\sam-3d-body-dinov3\assets\mhr_model.pt"
)

estimator = SAM3DBodyEstimator(
    sam_3d_body_model=model,
    model_cfg=model_cfg,
    human_detector=None,
    human_segmentor=None,
    fov_estimator=None,
)

# -----------------------
# Open webcam
# -----------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_vis = frame.copy()

    # -----------------------
    # Inference
    # -----------------------
    outputs = estimator.process_one_image(img_rgb)

    if outputs and len(outputs) > 0:

        # -----------------------
        # 2D keypoints (for drawing)
        # -----------------------
        keypoints_2d = outputs[0]["pred_keypoints_2d"]

        for x, y in keypoints_2d:
            cv2.circle(img_vis, (int(x), int(y)), 4, (0, 255, 0), -1)

        # -----------------------
        # 3D keypoints (for robot)
        # -----------------------
        keypoints_3d = outputs[0]["pred_keypoints_3d"]
        print("3D shape:", keypoints_3d.shape)

    # -----------------------
    # Show result
    # -----------------------
    cv2.imshow("SAM3D Webcam", img_vis)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()