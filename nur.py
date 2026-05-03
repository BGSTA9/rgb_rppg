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