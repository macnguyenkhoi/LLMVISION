import cv2
import requests
import json
import base64
import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk
import threading
import time
import numpy as np
from ultralytics import YOLO
import logging

class WebcamObjectDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Object Detection - Ollama + YOLO11")
        self.root.geometry("1000x700")
        
        # Make window slightly transparent
        self.root.attributes('-alpha', 0.99)
        
        # Ollama API settings
        self.ollama_url = "http://127.0.0.1:11434/api/generate"
        self.model_name = "gemma3:4b"
        
        # YOLO11 model settings
        self.yolo_model_path = r"C:\Users\khoimn1\Downloads\yolo11s.pt"
        self.yolo_model = None
        self.load_yolo_model()
        
        # Setup logging for detection times
        self.setup_logging()
        
        # Camera settings
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.detection_interval = 3  # seconds between detections
        self.last_detection_time = 0
        self.privacy_mode = True  # Enable privacy mode by default
        self.privacy_level = 0.6  # Higher value = more privacy overlay
        self.yolo_detections = []  # Store YOLO detection results for overlay
        
        # YOLO auto-detection settings
        self.yolo_auto_running = False
        self.yolo_fps = 5  # 5 frames per second
        self.yolo_interval = 1.0 / self.yolo_fps  # 0.2 seconds between detections
        self.last_yolo_detection_time = 0
        
        # Create GUI
        self.setup_gui()
        
        # Start camera
        self.start_camera()
        
    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Live Object Detection - Ollama + YOLO11", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # Video frame
        self.video_frame = ttk.LabelFrame(main_frame, text="Live Camera Feed", padding="5")
        self.video_frame.grid(row=1, column=0, padx=(0, 10), sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack()
        
        # Detection results frame
        results_frame = ttk.LabelFrame(main_frame, text="Detection Results", padding="5")
        results_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)
        
        # Status label
        self.status_label = ttk.Label(results_frame, text="Status: Starting camera...", 
                                     foreground="blue")
        self.status_label.grid(row=0, column=0, pady=(0, 10), sticky=tk.W)
        
        # Results text area
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, 
                                                     width=40, height=20)
        self.results_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Controls frame
        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=2, column=0, columnspan=2, pady=(10, 0))
        
        # Control buttons
        self.start_btn = ttk.Button(controls_frame, text="Start Detection", 
                                   command=self.toggle_detection)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.capture_btn = ttk.Button(controls_frame, text="Capture & Analyze", 
                                     command=self.manual_detection)
        self.capture_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Privacy mode toggle
        self.privacy_btn = ttk.Button(controls_frame, text="Privacy Mode: ON", 
                                     command=self.toggle_privacy_mode)
        self.privacy_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # YOLO11 detection button
        self.yolo_btn = ttk.Button(controls_frame, text="YOLO11 Detect", 
                                  command=self.yolo_detection)
        self.yolo_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # YOLO Auto-detection toggle
        self.yolo_auto_btn = ttk.Button(controls_frame, text="YOLO Auto: OFF", 
                                       command=self.toggle_yolo_auto)
        self.yolo_auto_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Clear detections button
        self.clear_btn = ttk.Button(controls_frame, text="Clear Overlay", 
                                   command=self.clear_detections)
        self.clear_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Interval setting
        ttk.Label(controls_frame, text="Detection Interval (seconds):").pack(side=tk.LEFT, padx=(20, 5))
        self.interval_var = tk.StringVar(value=str(self.detection_interval))
        interval_entry = ttk.Entry(controls_frame, textvariable=self.interval_var, width=5)
        interval_entry.pack(side=tk.LEFT)
        interval_entry.bind('<Return>', self.update_interval)
        
    def start_camera(self):
        """Initialize and start the camera"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Could not open camera")
                
            # Set camera resolution (smaller for privacy)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            
            self.status_label.config(text="Status: Camera started", foreground="green")
            self.update_video_feed()
            
        except Exception as e:
            self.status_label.config(text=f"Status: Camera error - {str(e)}", 
                                   foreground="red")
            
    def update_video_feed(self):
        """Update the video feed in the GUI"""
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                
                # Convert frame for display (smaller and more transparent)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Draw YOLO detections on the frame
                if self.yolo_detections:
                    frame_rgb = self.draw_yolo_detections(frame_rgb)
                
                frame_pil = Image.fromarray(frame_rgb)
                
                # Make image smaller and reduce opacity for privacy
                frame_pil = frame_pil.resize((240, 180), Image.Resampling.LANCZOS)
                
                # Apply privacy overlay if enabled
                if self.privacy_mode:
                    overlay = Image.new('RGBA', frame_pil.size, (128, 128, 128, 120))
                    frame_pil = frame_pil.convert('RGBA')
                    frame_pil = Image.blend(frame_pil, overlay, self.privacy_level)
                
                frame_tk = ImageTk.PhotoImage(frame_pil)
                
                self.video_label.configure(image=frame_tk)
                self.video_label.image = frame_tk
                
                # Auto-detection if enabled
                if self.is_running:
                    current_time = time.time()
                    if current_time - self.last_detection_time > self.detection_interval:
                        self.last_detection_time = current_time
                        threading.Thread(target=self.detect_objects, daemon=True).start()
                
                # YOLO auto-detection if enabled
                if self.yolo_auto_running:
                    current_time = time.time()
                    if current_time - self.last_yolo_detection_time > self.yolo_interval:
                        self.last_yolo_detection_time = current_time
                        threading.Thread(target=self.run_yolo_detection, daemon=True).start()
        
        # Schedule next update
        self.root.after(30, self.update_video_feed)
        
    def encode_image_to_base64(self, frame):
        """Convert OpenCV frame to base64 string"""
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            # Resize to reduce processing time
            pil_image = pil_image.resize((512, 384), Image.Resampling.LANCZOS)
            # Convert to bytes
            import io
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=85)
            image_bytes = buffer.getvalue()
            # Encode to base64
            base64_string = base64.b64encode(image_bytes).decode('utf-8')
            return base64_string
        except Exception as e:
            print(f"Error encoding image: {e}")
            return None
            
    def detect_objects(self):
        """Send frame to Ollama for object detection"""
        if self.current_frame is None:
            return
            
        try:
            # Update status
            self.root.after(0, lambda: self.status_label.config(
                text="Status: Analyzing image...", foreground="orange"))
            
            # Encode frame to base64
            base64_image = self.encode_image_to_base64(self.current_frame)
            if base64_image is None:
                return
                
            # Prepare request to Ollama
            payload = {
                "model": self.model_name,
                "prompt": "Analyze this image and list all the objects you can see. Be specific and descriptive. Format your response as a clear list of objects with their approximate locations or descriptions.",
                "images": [base64_image],
                "stream": False
            }
            
            # Send request
            response = requests.post(self.ollama_url, json=payload, timeout=100)
            
            if response.status_code == 200:
                result = response.json()
                detection_result = result.get('response', 'No response received')
                
                # Update results in GUI
                timestamp = time.strftime("%H:%M:%S")
                self.root.after(0, lambda: self.update_results(timestamp, detection_result))
                self.root.after(0, lambda: self.status_label.config(
                    text="Status: Detection complete", foreground="green"))
            else:
                error_msg = f"Error: HTTP {response.status_code}"
                self.root.after(0, lambda: self.update_results(
                    time.strftime("%H:%M:%S"), error_msg))
                self.root.after(0, lambda: self.status_label.config(
                    text="Status: Detection failed", foreground="red"))
                
        except requests.exceptions.RequestException as e:
            error_msg = f"Connection error: {str(e)}"
            self.root.after(0, lambda: self.update_results(
                time.strftime("%H:%M:%S"), error_msg))
            self.root.after(0, lambda: self.status_label.config(
                text="Status: Connection failed", foreground="red"))
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self.root.after(0, lambda: self.update_results(
                time.strftime("%H:%M:%S"), error_msg))
            self.root.after(0, lambda: self.status_label.config(
                text="Status: Error occurred", foreground="red"))
                
    def update_results(self, timestamp, result):
        """Update the results text area"""
        self.results_text.insert(tk.END, f"\n{'='*50}\n")
        self.results_text.insert(tk.END, f"Detection at {timestamp}:\n")
        self.results_text.insert(tk.END, f"{result}\n")
        self.results_text.see(tk.END)
        
    def toggle_detection(self):
        """Toggle automatic detection on/off"""
        self.is_running = not self.is_running
        if self.is_running:
            self.start_btn.config(text="Stop Detection")
            self.status_label.config(text="Status: Auto-detection enabled", foreground="green")
        else:
            self.start_btn.config(text="Start Detection")
            self.status_label.config(text="Status: Auto-detection disabled", foreground="blue")
            
    def manual_detection(self):
        """Trigger manual detection"""
        if self.current_frame is not None:
            threading.Thread(target=self.detect_objects, daemon=True).start()
            
    def toggle_privacy_mode(self):
        """Toggle privacy mode on/off"""
        self.privacy_mode = not self.privacy_mode
        if self.privacy_mode:
            self.privacy_btn.config(text="Privacy Mode: ON")
            self.status_label.config(text="Status: Privacy mode enabled", foreground="blue")
        else:
            self.privacy_btn.config(text="Privacy Mode: OFF")
            self.status_label.config(text="Status: Privacy mode disabled", foreground="orange")
            
    def update_interval(self, event=None):
        """Update detection interval"""
        try:
            new_interval = float(self.interval_var.get())
            if new_interval > 0:
                self.detection_interval = new_interval
                self.status_label.config(text=f"Status: Interval updated to {new_interval}s", 
                                       foreground="blue")
        except ValueError:
            self.interval_var.set(str(self.detection_interval))
    
    def load_yolo_model(self):
        """Load YOLO11 model"""
        try:
            self.yolo_model = YOLO(self.yolo_model_path)
            print("YOLO11 model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLO11 model: {e}")
            self.yolo_model = None
    
    def setup_logging(self):
        """Setup logging for detection times"""
        # Create logs directory if it doesn't exist
        import os
        log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logger
        self.detection_logger = logging.getLogger('yolo_detection')
        self.detection_logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = os.path.join(log_dir, 'yolo_detection_times.log')
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Add handler to logger
        if not self.detection_logger.handlers:
            self.detection_logger.addHandler(handler)
    
    def yolo_detection(self):
        """Run YOLO11 detection on current frame"""
        if self.current_frame is None:
            self.status_label.config(text="Status: No frame available", foreground="red")
            return
            
        if self.yolo_model is None:
            self.status_label.config(text="Status: YOLO11 model not loaded", foreground="red")
            return
            
        threading.Thread(target=self.run_yolo_detection, daemon=True).start()
    
    def toggle_yolo_auto(self):
        """Toggle YOLO auto-detection at 5 FPS"""
        self.yolo_auto_running = not self.yolo_auto_running
        if self.yolo_auto_running:
            self.yolo_auto_btn.config(text="YOLO Auto: ON")
            self.status_label.config(text=f"Status: YOLO auto-detection enabled ({self.yolo_fps} FPS)", 
                                   foreground="green")
            # Clear previous detections and reset timer
            self.yolo_detections = []
            self.last_yolo_detection_time = 0
        else:
            self.yolo_auto_btn.config(text="YOLO Auto: OFF")
            self.status_label.config(text="Status: YOLO auto-detection disabled", foreground="blue")
    
    def run_yolo_detection(self):
        """Run YOLO11 detection in separate thread"""
        # Check if model is available and frame exists
        if self.yolo_model is None or self.current_frame is None:
            return
            
        try:
            # Start timing
            start_time = time.time()
            start_time_ms = int(start_time * 1000)
            
            # Update status (only if not in auto mode to avoid UI spam)
            if not self.yolo_auto_running:
                self.root.after(0, lambda: self.status_label.config(
                    text="Status: Running YOLO11 detection...", foreground="orange"))
            
            # Run YOLO11 detection
            detection_start = time.time()
            results = self.yolo_model(self.current_frame)
            detection_end = time.time()
            
            # Calculate detection time in milliseconds
            detection_time_ms = int((detection_end - detection_start) * 1000)
            
            # Clear previous detections
            self.yolo_detections = []
            
            # Parse results and store for overlay
            detection_text = "YOLO11 Detection Results:\n"
            detection_text += "=" * 30 + "\n"
            detection_text += f"Detection Time: {detection_time_ms}ms\n"
            detection_text += "-" * 20 + "\n"
            
            objects_detected = 0
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class name and confidence
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = self.yolo_model.names[class_id]
                        
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Store detection for overlay
                        self.yolo_detections.append({
                            'class_name': class_name,
                            'confidence': confidence,
                            'bbox': (int(x1), int(y1), int(x2), int(y2))
                        })
                        
                        detection_text += f"Object: {class_name}\n"
                        detection_text += f"Confidence: {confidence:.2f}\n"
                        detection_text += f"Location: ({int(x1)}, {int(y1)}) to ({int(x2)}, {int(y2)})\n"
                        detection_text += "-" * 20 + "\n"
                        objects_detected += 1
                else:
                    detection_text += "No objects detected\n"
            
            # Calculate total processing time
            total_time_ms = int((time.time() - start_time) * 1000)
            
            # Log detection time and results
            mode_indicator = "AUTO" if self.yolo_auto_running else "MANUAL"
            log_message = f"YOLO Detection ({mode_indicator}) - Time: {detection_time_ms}ms, Total Processing: {total_time_ms}ms, Objects: {objects_detected}"
            self.detection_logger.info(log_message)
            
            # Only print to console if not in auto mode to avoid spam
            if not self.yolo_auto_running:
                print(f"[LOG] {log_message}")
            
            # Update results in GUI (only for manual detection to avoid spam)
            if not self.yolo_auto_running:
                timestamp = time.strftime("%H:%M:%S")
                self.root.after(0, lambda: self.update_results(timestamp, detection_text))
                self.root.after(0, lambda: self.status_label.config(
                    text=f"Status: YOLO11 detection complete ({detection_time_ms}ms)", foreground="green"))
                
        except Exception as e:
            error_msg = f"YOLO11 detection error: {str(e)}"
            self.detection_logger.error(f"YOLO Detection Error: {str(e)}")
            if not self.yolo_auto_running:
                self.root.after(0, lambda: self.update_results(
                    time.strftime("%H:%M:%S"), error_msg))
                self.root.after(0, lambda: self.status_label.config(
                    text="Status: YOLO11 detection failed", foreground="red"))
    
    def draw_yolo_detections(self, frame_rgb):
        """Draw YOLO detection results on the frame"""
        try:
            import cv2
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            for detection in self.yolo_detections:
                class_name = detection['class_name']
                confidence = detection['confidence']
                x1, y1, x2, y2 = detection['bbox']
                
                # Draw bounding box
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Prepare label text
                label = f"{class_name}: {confidence:.2f}"
                
                # Get text size for background rectangle
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                
                # Draw background rectangle for text
                cv2.rectangle(frame_bgr, (x1, y1 - text_height - 10), 
                             (x1 + text_width, y1), (0, 255, 0), -1)
                
                # Draw text
                cv2.putText(frame_bgr, label, (x1, y1 - 5), 
                           font, font_scale, (0, 0, 0), thickness)
            
            # Convert back to RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
        except Exception as e:
            print(f"Error drawing detections: {e}")
        
        return frame_rgb
    
    def clear_detections(self):
        """Clear YOLO detection overlays"""
        self.yolo_detections = []
        self.status_label.config(text="Status: Detection overlay cleared", foreground="blue")
            
    def cleanup(self):
        """Clean up resources"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        
    def on_closing(self):
        """Handle window closing"""
        self.cleanup()
        self.root.destroy()

def main():
    # Check if Ollama is running
    try:
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
        if response.status_code != 200:
            print("Warning: Ollama server might not be running on localhost:11434")
    except requests.exceptions.RequestException:
        print("Warning: Cannot connect to Ollama server. Make sure it's running with 'ollama serve'")
    
    # Create and run the application
    root = tk.Tk()
    app = WebcamObjectDetector(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
