import cv2
import threading
import time
from core_engine import DepthConditionedYOLO # Imports your finalized architecture

class ThreadedCamera:
    def __init__(self, src=0):
        """
        Initializes a background thread to constantly read the RTSP/Webcam stream,
        preventing OpenCV buffer lag and ensuring the AI only processes the freshest frame.
        """
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        
        # Read the first frame to establish the connection
        self.status, self.frame = self.capture.read()
        
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.capture.read()
            with self.read_lock:
                self.status = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            # Return a copy to prevent the background thread from overwriting 
            # the frame while the AI is currently processing it
            frame = self.frame.copy() 
        return self.status, frame

    def stop(self):
        self.started = False
        self.thread.join()
        self.capture.release()

if __name__ == "__main__":
    print("🚀 Booting Live Stream Intrusion Pipeline...")
    
    # 1. Initialize the AI Engine
    engine = DepthConditionedYOLO()
    
    # 2. Connect to Camera
    # Use 0 for your laptop webcam, or paste your RTSP link: "rtsp://admin:password@192.168.1.X:554/stream"
    CAMERA_SOURCE = 2
    
    print(f"📡 Connecting to stream: {CAMERA_SOURCE}")
    cam = ThreadedCamera(src=CAMERA_SOURCE).start()
    time.sleep(1.0) # Allow the sensor to warm up
    
    print("✅ Live Stream Active. Press 'q' to quit.")

    while True:
        status, frame = cam.read()
        if not status:
            continue

        start_time = time.perf_counter()

        # Run the Dual-Gate AI Pipeline
        predictions = engine.predict(frame)

        # Calculate FPS
        fps = 1.0 / (time.perf_counter() - start_time)

        # --- Draw Visualization ---
        for pred in predictions:
            x1, y1, x2, y2 = pred["box"]
            is_real = pred["is_real_3d"]
            
            # Green for Real Humans, Red for 2D Spoofs/Reflections
            color = (0, 255, 0) if is_real else (0, 0, 255)
            label = "Real Human" if is_real else "2D Spoof"
            
            # Draw Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw Label Background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
            
            # Draw Text
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # Display FPS telemetry on screen
        cv2.putText(frame, f"Edge AI FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Show the live feed
        cv2.imshow("Zero-Shot Anti-Spoofing Engine", frame)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cam.stop()
    cv2.destroyAllWindows()
