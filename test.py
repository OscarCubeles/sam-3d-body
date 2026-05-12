import cv2
import numpy as np
import torch

from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator


# -----------------------
# Load model ONLY
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# Run inference
# -----------------------
img_bgr = cv2.imread(r"C:\Users\oscar\Documents\TU\TFM\sam-3d-body\man_woman.jpg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

img_vis = img_bgr.copy()

outputs = estimator.process_one_image(img_rgb)

if outputs and len(outputs) > 0:
    keypoints_2d = outputs[0]["pred_keypoints_2d"]

    for x, y in keypoints_2d:
        cv2.circle(img_vis, (int(x), int(y)), 4, (0, 255, 0), -1)

cv2.imshow("result", img_vis)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -----------------------
# Extract 3D keypoints ONLY
# -----------------------
if outputs and len(outputs) > 0:
    keypoints_3d = outputs[0]["pred_keypoints_3d"]
    print(keypoints_3d)