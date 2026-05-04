# 1. Real-Time Inference
import rppg
import time
import cv2

model = rppg.Model()
# Open the default camera (index 0)
with model.video_capture(0):
    last_process_time = 0
    current_hr = None
    
    # Iterate through the preview generator (this is the main loop)
    for frame, box in model.preview:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # 1. Calculate HR every 1 second to avoid lag
        now = time.time()
        if now - last_process_time > 1.0:
            result = model.hr(start=-10)
            if result and result['hr']:
                current_hr = result['hr']
                print(f"Real-time HR: {current_hr:.1f} BPM")
            last_process_time = now
            
        # 2. Visualization
        if box is not None:
            # box format: [[row_min, row_max], [col_min, col_max]]
            y1, y2 = box[0]
            x1, x2 = box[1]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Display HR on the frame if available
            if current_hr is not None:
                cv2.putText(frame, f"HR: {current_hr:.1f}", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow("rPPG Monitor", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 2. Advanced API

# --> Retrieving Raw Signals
# Retrieve the full BVP signal and corresponding timestamps
bvp, timestamps = model.bvp()
# Retrieve the raw, unfiltered BVP signal
raw_bvp, timestamps = model.bvp(raw=True)

# --> Time Slicing
# Get signal from t=10s to t=20s
bvp_slice, ts_slice = model.bvp(start=10, end=20)
# Get metrics for the last 15 seconds
metrics = model.hr(start=-15)

# --> Tensor Inputs
import numpy as np
# tensor shape: (T, H, W, 3)
video_tensor = np.zeros((300, 480, 640, 3), dtype='uint8') # 480p video
result = model.process_video_tensor(video_tensor, fps=30.0)
faces_tensor = np.zeros((300, 128, 128, 3), dtype='uint8') # face array
result = model.process_faces_tensor(faces_tensor, fps=30.0)

# --> Model Selection
# Example: Initialize the PhysMamba model
model = rppg.Model('PhysMamba.pure')

# Model Zoo
# ME-chunk -> State-space model rPPG (chunk inference) -> arXiv 2025
# ME-flow -> State-space model rPPG (low-latency flow) -> arXiv 2025
# PhysMamba -> Dual-branch Mamba architecture -> CCBR 2024
# RhythmMamba -> Frequency-domain constrained Mamba -> AAAI 2025
# PhysFormer -> Temporal Difference Transformer -> CVPR 2022
# TSCAN -> Temporal Shift Convolutional Attention Network -> NeurIPS 2020
# EfficientPhys -> Self-attention variant of TSCAN -> WACV 2023
# PhysNet -> 3D Convolutional Encoder-Decoder -> BMVC 2019
# FacePhys -> Optimized state-space model -> - 

