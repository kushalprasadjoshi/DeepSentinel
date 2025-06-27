import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time
from deep_sentinel import VideoPipeline, CameraManager, SystemState
from deep_sentinel.utils import logging_utils

logger = logging_utils.setup_module_logger(__name__)

class MainApplication(tk.Tk):
    """Main application window for DeepSentinel
    
    Attributes:
        config: Application configuration
        camera_manager: CameraManager instance
        state_manager: SystemState instance
        video_pipeline: VideoPipeline instance
        video_label: Label for video display
        threat_tree: Treeview for threat display
        status_bar: Status bar widget
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.title("DeepSentinel Security System")
        self.geometry("1200x800")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Initialize core components
        self.camera_manager = CameraManager(config['camera'])
        self.state_manager = SystemState(config)
        self.video_pipeline = VideoPipeline(
            self.camera_manager,
            self.state_manager,
            config
        )
        
        # Create GUI layout
        self.create_widgets()
        
        # Start video pipeline
        self.video_pipeline.start()
        
        # Start UI update loop
        self.update_ui()
    
    def create_widgets(self):
        """Create and arrange GUI widgets"""
        # Configure grid layout
        self.grid_columnconfigure(0, weight=2)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=3)
        self.grid_rowconfigure(1, weight=1)
        
        # Video display panel
        video_frame = ttk.LabelFrame(self, text="Live Feed")
        video_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(fill="both", expand=True)
        
        # Threat information panel
        threat_frame = ttk.LabelFrame(self, text="Threat Analysis")
        threat_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        # Threat treeview
        self.threat_tree = ttk.Treeview(threat_frame, columns=("confidence", "time", "camera"))
        self.threat_tree.heading("#0", text="Threat Type")
        self.threat_tree.heading("confidence", text="Confidence")
        self.threat_tree.heading("time", text="Time")
        self.threat_tree.heading("camera", text="Camera")
        self.threat_tree.column("#0", width=150)
        self.threat_tree.column("confidence", width=80)
        self.threat_tree.column("time", width=120)
        self.threat_tree.column("camera", width=100)
        self.threat_tree.pack(fill="both", expand=True)
        
        # Control panel
        control_frame = ttk.Frame(self)
        control_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        
        # Add control buttons
        ttk.Button(control_frame, text="Start Monitoring", command=self.start_monitoring).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Pause", command=self.pause_monitoring).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Settings", command=self.open_settings).pack(side="left", padx=5)
        ttk.Button(control_frame, text="View Alerts", command=self.view_alerts).pack(side="left", padx=5)
        
        # Status bar
        self.status_bar = ttk.Label(self, text="System Initializing...", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=2, column=0, columnspan=2, sticky="ew")
    
    def update_ui(self):
        """Update the UI components"""
        # Update video feed
        frame = self.video_pipeline.get_current_frame()
        if frame is not None:
            # Convert to PhotoImage
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = img.resize((800, 600))
            self.photo = ImageTk.PhotoImage(image=img)
            self.video_label.config(image=self.photo)
        
        # Update threat list
        recent_threats = self.state_manager.get_recent_threats(10)
        self.update_threat_tree(recent_threats)
        
        # Update status bar
        fps = self.video_pipeline.get_fps()
        status_text = f"FPS: {fps:.1f} | Threats: {self.state_manager.system_status['detection_count']}"
        self.status_bar.config(text=status_text)
        
        # Schedule next update
        self.after(50, self.update_ui)
    
    def update_threat_tree(self, threats):
        """Update threat treeview with new threats"""
        # Clear existing items
        for item in self.threat_tree.get_children():
            self.threat_tree.delete(item)
        
        # Add new threats
        for threat in threats:
            time_str = time.strftime('%H:%M:%S', time.localtime(threat['timestamp']))
            self.threat_tree.insert("", "end", text=threat['type'], 
                                   values=(f"{threat['confidence']:.2f}", time_str, threat['location']))
    
    def start_monitoring(self):
        """Start monitoring process"""
        self.video_pipeline.start()
        logger.info("Monitoring started")
    
    def pause_monitoring(self):
        """Pause monitoring process"""
        self.video_pipeline.stop()
        logger.info("Monitoring paused")
    
    def open_settings(self):
        """Open settings dialog"""
        # Placeholder for settings implementation
        logger.info("Settings dialog opened")
    
    def view_alerts(self):
        """View alert history"""
        # Placeholder for alert viewer
        logger.info("Alert history viewed")
    
    def on_close(self):
        """Handle application close"""
        self.video_pipeline.stop()
        self.camera_manager.release()
        self.destroy()
        logger.info("Application closed")