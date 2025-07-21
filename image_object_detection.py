import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
from PIL import Image, ImageTk
import requests
import json
import base64
import io
import threading
import time
import os
import logging

class ImageObjectDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Object Detection - Gemma3:4b Local")
        self.root.geometry("1200x800")
        
        # Ollama API settings for local Gemma3:4b
        self.ollama_url = "http://127.0.0.1:11434/api/generate"
        self.model_name = "gemma3:4b"
        
        # Current image
        self.current_image = None
        self.current_image_path = None
        
        # Detection history
        self.detection_history = []
        
        # Setup logging for detection times
        self.setup_logging()
        
        # Create GUI
        self.setup_gui()
        
        # Check Ollama connection on startup
        self.check_ollama_connection()
        
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
        title_label = ttk.Label(main_frame, text="Image Object Detection - Gemma3:4b Local", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # Image display frame
        self.image_frame = ttk.LabelFrame(main_frame, text="Image Preview", padding="5")
        self.image_frame.grid(row=1, column=0, padx=(0, 10), sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Image canvas with scrollbars
        canvas_frame = ttk.Frame(self.image_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.image_canvas = tk.Canvas(canvas_frame, bg="lightgray", width=500, height=400)
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.image_canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.image_canvas.xview)
        self.image_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        self.image_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        
        # Default message on canvas
        self.image_canvas.create_text(250, 200, text="No image loaded\nClick 'Upload Image' to start", 
                                     font=("Arial", 12), fill="gray")
        
        # Detection results frame
        results_frame = ttk.LabelFrame(main_frame, text="Detection Results", padding="5")
        results_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)
        
        # Status label
        self.status_label = ttk.Label(results_frame, text="Status: Ready to analyze images", 
                                     foreground="blue")
        self.status_label.grid(row=0, column=0, pady=(0, 10), sticky=tk.W)
        
        # Results text area
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, 
                                                     width=50, height=25)
        self.results_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Controls frame
        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=2, column=0, columnspan=2, pady=(10, 0))
        
        # Control buttons
        self.upload_btn = ttk.Button(controls_frame, text="📁 Upload Image", 
                                    command=self.upload_image, style="Accent.TButton")
        self.upload_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.analyze_btn = ttk.Button(controls_frame, text="🔍 Analyze Objects", 
                                     command=self.analyze_image, state="disabled")
        self.analyze_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.detailed_btn = ttk.Button(controls_frame, text="📋 Detailed Analysis", 
                                      command=self.detailed_analysis, state="disabled")
        self.detailed_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.clear_btn = ttk.Button(controls_frame, text="🗑️ Clear Results", 
                                   command=self.clear_results)
        self.clear_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.save_btn = ttk.Button(controls_frame, text="💾 Save Results", 
                                  command=self.save_results, state="disabled")
        self.save_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Separator
        separator = ttk.Separator(controls_frame, orient='vertical')
        separator.pack(side=tk.LEFT, fill=tk.Y, padx=(20, 20))
        
        # Analysis options
        self.analysis_options = ttk.LabelFrame(controls_frame, text="Analysis Options", padding="5")
        self.analysis_options.pack(side=tk.LEFT, padx=(0, 10))
        
        # Confidence level
        ttk.Label(self.analysis_options, text="Detail Level:").grid(row=0, column=0, sticky=tk.W)
        self.detail_var = tk.StringVar(value="Standard")
        detail_combo = ttk.Combobox(self.analysis_options, textvariable=self.detail_var, 
                                   values=["Basic", "Standard", "Detailed", "Comprehensive"], 
                                   width=12, state="readonly")
        detail_combo.grid(row=0, column=1, padx=(5, 0))
        
        # Include location info
        self.include_location = tk.BooleanVar(value=True)
        location_check = ttk.Checkbutton(self.analysis_options, text="Include positions", 
                                        variable=self.include_location)
        location_check.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        
    def setup_logging(self):
        """Setup logging for detection times"""
        # Create logs directory if it doesn't exist
        log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logger
        self.detection_logger = logging.getLogger('gemma_detection')
        self.detection_logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = os.path.join(log_dir, 'gemma_detection_times.log')
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Add handler to logger
        if not self.detection_logger.handlers:
            self.detection_logger.addHandler(handler)
    
    def check_ollama_connection(self):
        """Check if Ollama server is running and Gemma3:4b is available"""
        try:
            # Check if Ollama is running
            response = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json()
                model_names = [model['name'] for model in models.get('models', [])]
                
                if any('gemma3:4b' in name for name in model_names):
                    self.status_label.config(text="Status: Connected to Ollama - Gemma3:4b ready", 
                                           foreground="green")
                else:
                    self.status_label.config(text="Status: Gemma3:4b not found - run 'ollama pull gemma3:4b'", 
                                           foreground="orange")
                    messagebox.showwarning("Model Not Found", 
                                         "Gemma3:4b model not found. Please run:\nollama pull gemma3:4b")
            else:
                self.status_label.config(text="Status: Ollama server error", foreground="red")
        except requests.exceptions.RequestException:
            self.status_label.config(text="Status: Cannot connect to Ollama - run 'ollama serve'", 
                                   foreground="red")
            messagebox.showerror("Connection Error", 
                               "Cannot connect to Ollama server.\nPlease make sure Ollama is running with 'ollama serve'")
    
    def upload_image(self):
        """Upload and display an image"""
        file_types = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff *.webp"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select an image for object detection",
            filetypes=file_types
        )
        
        if file_path:
            try:
                # Load and display image
                self.current_image = Image.open(file_path)
                self.current_image_path = file_path
                
                # Display image on canvas
                self.display_image()
                
                # Update status and enable buttons
                self.status_label.config(text=f"Status: Image loaded - {os.path.basename(file_path)}", 
                                       foreground="green")
                self.analyze_btn.config(state="normal")
                self.detailed_btn.config(state="normal")
                
                # Clear previous results
                self.results_text.delete(1.0, tk.END)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
                self.status_label.config(text="Status: Failed to load image", foreground="red")
    
    def display_image(self):
        """Display the current image on the canvas"""
        if self.current_image:
            # Calculate display size (fit to canvas while maintaining aspect ratio)
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            
            # Use default size if canvas not yet rendered
            if canvas_width <= 1:
                canvas_width, canvas_height = 500, 400
            
            img_width, img_height = self.current_image.size
            
            # Calculate scaling factor
            scale_w = canvas_width / img_width
            scale_h = canvas_height / img_height
            scale = min(scale_w, scale_h, 1.0)  # Don't upscale
            
            # Resize image for display
            display_width = int(img_width * scale)
            display_height = int(img_height * scale)
            
            display_image = self.current_image.resize((display_width, display_height), 
                                                    Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            self.photo = ImageTk.PhotoImage(display_image)
            
            # Clear canvas and display image
            self.image_canvas.delete("all")
            self.image_canvas.create_image(canvas_width//2, canvas_height//2, 
                                         image=self.photo, anchor=tk.CENTER)
            
            # Update scroll region
            self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all"))
    
    def encode_image_to_base64(self, image):
        """Convert PIL Image to base64 string"""
        try:
            # Resize image if too large (to reduce processing time and API payload)
            max_size = 1024
            img_width, img_height = image.size
            
            if max(img_width, img_height) > max_size:
                if img_width > img_height:
                    new_width = max_size
                    new_height = int(img_height * max_size / img_width)
                else:
                    new_height = max_size
                    new_width = int(img_width * max_size / img_height)
                
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to bytes
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=85)
            image_bytes = buffer.getvalue()
            
            # Encode to base64
            base64_string = base64.b64encode(image_bytes).decode('utf-8')
            return base64_string
        except Exception as e:
            print(f"Error encoding image: {e}")
            return None
    
    def analyze_image(self):
        """Analyze the current image for objects"""
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return
        
        # Disable button during analysis
        self.analyze_btn.config(state="disabled")
        
        # Run analysis in separate thread
        threading.Thread(target=self._run_analysis, args=("standard",), daemon=True).start()
    
    def detailed_analysis(self):
        """Run detailed analysis on the current image"""
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return
        
        # Disable button during analysis
        self.detailed_btn.config(state="disabled")
        
        # Run detailed analysis in separate thread
        threading.Thread(target=self._run_analysis, args=("detailed",), daemon=True).start()
    
    def _run_analysis(self, analysis_type="standard"):
        """Run object detection analysis"""
        try:
            start_time = time.time()
            
            # Update status
            self.root.after(0, lambda: self.status_label.config(
                text="Status: Analyzing image with Gemma3:4b...", foreground="orange"))
            
            # Encode image to base64
            base64_image = self.encode_image_to_base64(self.current_image)
            if base64_image is None:
                raise Exception("Failed to encode image")
            
            # Prepare prompt based on analysis type and options
            prompt = self._create_analysis_prompt(analysis_type)
            
            # Prepare request to Ollama
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [base64_image],
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Lower temperature for more consistent results
                    "top_p": 0.9
                }
            }
            
            # Send request
            response = requests.post(self.ollama_url, json=payload, timeout=300)
            
            if response.status_code == 200:
                result = response.json()
                detection_result = result.get('response', 'No response received')
                
                # Calculate processing time
                processing_time = int((time.time() - start_time) * 1000)
                
                # Log the detection
                log_message = f"Gemma3:4b Analysis ({analysis_type.upper()}) - Time: {processing_time}ms, Image: {os.path.basename(self.current_image_path or 'Unknown')}"
                self.detection_logger.info(log_message)
                print(f"[LOG] {log_message}")
                
                # Store in history
                self.detection_history.append({
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'image_path': self.current_image_path,
                    'analysis_type': analysis_type,
                    'processing_time': processing_time,
                    'result': detection_result
                })
                
                # Update results in GUI
                timestamp = time.strftime("%H:%M:%S")
                self.root.after(0, lambda: self._update_results(timestamp, detection_result, analysis_type, processing_time))
                self.root.after(0, lambda: self.status_label.config(
                    text=f"Status: Analysis complete ({processing_time}ms)", foreground="green"))
                self.root.after(0, lambda: self.save_btn.config(state="normal"))
            else:
                error_msg = f"Error: HTTP {response.status_code} - {response.text}"
                self.root.after(0, lambda: self._update_results(
                    time.strftime("%H:%M:%S"), error_msg, analysis_type))
                self.root.after(0, lambda: self.status_label.config(
                    text="Status: Analysis failed", foreground="red"))
                
        except requests.exceptions.RequestException as e:
            error_msg = f"Connection error: {str(e)}"
            self.root.after(0, lambda: self._update_results(
                time.strftime("%H:%M:%S"), error_msg, analysis_type))
            self.root.after(0, lambda: self.status_label.config(
                text="Status: Connection failed", foreground="red"))
        except Exception as e:
            error_msg = f"Analysis error: {str(e)}"
            self.root.after(0, lambda: self._update_results(
                time.strftime("%H:%M:%S"), error_msg, analysis_type))
            self.root.after(0, lambda: self.status_label.config(
                text="Status: Analysis failed", foreground="red"))
        finally:
            # Re-enable buttons
            self.root.after(0, lambda: self.analyze_btn.config(state="normal"))
            self.root.after(0, lambda: self.detailed_btn.config(state="normal"))
    
    def _create_analysis_prompt(self, analysis_type):
        """Create analysis prompt based on type and options"""
        detail_level = self.detail_var.get()
        include_location = self.include_location.get()
        
        base_prompt = "Analyze this image and identify all objects you can see."
        
        if analysis_type == "detailed":
            base_prompt = "Perform a comprehensive analysis of this image and identify all objects, people, animals, and items you can see."
        
        # Add detail level instructions
        if detail_level == "Basic":
            detail_instruction = " Provide a simple list of the main objects."
        elif detail_level == "Standard":
            detail_instruction = " Be specific and descriptive about each object."
        elif detail_level == "Detailed":
            detail_instruction = " Provide detailed descriptions including colors, sizes, and characteristics of each object."
        else:  # Comprehensive
            detail_instruction = " Provide comprehensive descriptions including colors, sizes, materials, conditions, and any notable details about each object."
        
        # Add location instructions
        if include_location:
            location_instruction = " Include approximate locations or positions of objects in the image (like 'top-left', 'center', 'bottom-right', 'foreground', 'background')."
        else:
            location_instruction = ""
        
        # Combine into final prompt
        prompt = base_prompt + detail_instruction + location_instruction + " Format your response as a clear, organized list."
        
        return prompt
    
    def _update_results(self, timestamp, result, analysis_type, processing_time=None):
        """Update the results text area"""
        self.results_text.insert(tk.END, f"\n{'='*60}\n")
        self.results_text.insert(tk.END, f"Analysis at {timestamp} ({analysis_type.upper()})")
        if processing_time:
            self.results_text.insert(tk.END, f" - {processing_time}ms")
        self.results_text.insert(tk.END, f"\n")
        if self.current_image_path:
            self.results_text.insert(tk.END, f"Image: {os.path.basename(self.current_image_path)}\n")
        self.results_text.insert(tk.END, f"{'='*60}\n")
        self.results_text.insert(tk.END, f"{result}\n")
        self.results_text.see(tk.END)
    
    def clear_results(self):
        """Clear all results"""
        self.results_text.delete(1.0, tk.END)
        self.status_label.config(text="Status: Results cleared", foreground="blue")
        self.save_btn.config(state="disabled")
    
    def save_results(self):
        """Save analysis results to a file"""
        if not self.detection_history:
            messagebox.showinfo("No Results", "No analysis results to save.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Analysis Results",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("Image Object Detection Results - Gemma3:4b\n")
                    f.write("=" * 50 + "\n\n")
                    
                    for i, entry in enumerate(self.detection_history, 1):
                        f.write(f"Analysis #{i}\n")
                        f.write(f"Timestamp: {entry['timestamp']}\n")
                        f.write(f"Image: {os.path.basename(entry['image_path']) if entry['image_path'] else 'Unknown'}\n")
                        f.write(f"Analysis Type: {entry['analysis_type'].upper()}\n")
                        f.write(f"Processing Time: {entry['processing_time']}ms\n")
                        f.write("-" * 30 + "\n")
                        f.write(f"{entry['result']}\n")
                        f.write("\n" + "=" * 50 + "\n\n")
                
                messagebox.showinfo("Saved", f"Results saved to:\n{file_path}")
                self.status_label.config(text="Status: Results saved successfully", foreground="green")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results:\n{str(e)}")

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
    
    # Configure style
    style = ttk.Style()
    style.theme_use('clam')
    
    app = ImageObjectDetector(root)
    root.mainloop()

if __name__ == "__main__":
    main()
