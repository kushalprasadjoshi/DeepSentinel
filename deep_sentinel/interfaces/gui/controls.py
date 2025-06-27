import tkinter as tk
from tkinter import ttk
from deep_sentinel.utils import logging_utils

logger = logging_utils.setup_module_logger(__name__)

class ControlPanel(ttk.Frame):
    """Control panel for system settings and operations
    
    Attributes:
        camera_manager: CameraManager instance
        state_manager: SystemState instance
        video_pipeline: VideoPipeline instance
    """
    
    def __init__(self, parent, camera_manager, state_manager, video_pipeline):
        """
        Initialize control panel
        
        Args:
            parent: Parent widget
            camera_manager: CameraManager instance
            state_manager: SystemState instance
            video_pipeline: VideoPipeline instance
        """
        super().__init__(parent)
        self.camera_manager = camera_manager
        self.state_manager = state_manager
        self.video_pipeline = video_pipeline
        
        # Create tabs for different control sections
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.camera_tab = ttk.Frame(self.notebook)
        self.detection_tab = ttk.Frame(self.notebook)
        self.alerts_tab = ttk.Frame(self.notebook)
        self.system_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.camera_tab, text="Cameras")
        self.notebook.add(self.detection_tab, text="Detection")
        self.notebook.add(self.alerts_tab, text="Alerts")
        self.notebook.add(self.system_tab, text="System")
        
        # Setup each tab
        self.setup_camera_tab()
        self.setup_detection_tab()
        self.setup_alerts_tab()
        self.setup_system_tab()
    
    def setup_camera_tab(self):
        """Setup camera controls"""
        # Camera selection
        camera_frame = ttk.LabelFrame(self.camera_tab, text="Camera Selection")
        camera_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(camera_frame, text="Select Camera:").grid(row=0, column=0, padx=5, pady=5)
        
        self.camera_var = tk.StringVar()
        self.camera_dropdown = ttk.Combobox(camera_frame, textvariable=self.camera_var)
        self.camera_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        # Populate cameras
        self.refresh_camera_list()
        
        # Camera controls
        ttk.Button(camera_frame, text="Refresh List", command=self.refresh_camera_list).grid(row=0, column=2, padx=5)
        ttk.Button(camera_frame, text="Switch Camera", command=self.switch_camera).grid(row=0, column=3, padx=5)
        
        # Camera settings
        settings_frame = ttk.LabelFrame(self.camera_tab, text="Camera Settings")
        settings_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(settings_frame, text="Resolution:").grid(row=0, column=0, padx=5, pady=5)
        self.res_var = tk.StringVar(value="1280x720")
        ttk.Combobox(settings_frame, textvariable=self.res_var, 
                    values=["640x480", "1280x720", "1920x1080"]).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(settings_frame, text="FPS:").grid(row=1, column=0, padx=5, pady=5)
        self.fps_var = tk.IntVar(value=30)
        ttk.Spinbox(settings_frame, textvariable=self.fps_var, from_=1, to=60).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Button(settings_frame, text="Apply Settings", command=self.apply_camera_settings).grid(row=2, column=1, pady=10)
    
    def setup_detection_tab(self):
        """Setup detection controls"""
        # Sensitivity controls
        sens_frame = ttk.LabelFrame(self.detection_tab, text="Detection Sensitivity")
        sens_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(sens_frame, text="Motion Sensitivity:").grid(row=0, column=0, padx=5, pady=5)
        self.motion_sens = tk.IntVar(value=50)
        ttk.Scale(sens_frame, from_=0, to=100, variable=self.motion_sens, 
                 orient=tk.HORIZONTAL).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        ttk.Label(sens_frame, text="Threat Confidence:").grid(row=1, column=0, padx=5, pady=5)
        self.confidence_thresh = tk.DoubleVar(value=0.75)
        ttk.Scale(sens_frame, from_=0.1, to=1.0, variable=self.confidence_thresh, 
                 orient=tk.HORIZONTAL).grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        # Detection toggles
        toggle_frame = ttk.LabelFrame(self.detection_tab, text="Detection Features")
        toggle_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.weapon_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(toggle_frame, text="Weapon Detection", variable=self.weapon_var).pack(anchor=tk.W, padx=5, pady=2)
        
        self.intruder_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(toggle_frame, text="Intruder Detection", variable=self.intruder_var).pack(anchor=tk.W, padx=5, pady=2)
        
        self.ppe_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(toggle_frame, text="PPE Detection", variable=self.ppe_var).pack(anchor=tk.W, padx=5, pady=2)
        
        self.motion_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(toggle_frame, text="Motion Detection", variable=self.motion_var).pack(anchor=tk.W, padx=5, pady=2)
        
        ttk.Button(toggle_frame, text="Apply Settings", command=self.apply_detection_settings).pack(pady=10)
    
    def setup_alerts_tab(self):
        """Setup alert controls"""
        # Alert channels
        channel_frame = ttk.LabelFrame(self.alerts_tab, text="Alert Channels")
        channel_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.email_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(channel_frame, text="Email Alerts", variable=self.email_var).pack(anchor=tk.W, padx=5, pady=2)
        
        self.sms_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(channel_frame, text="SMS Alerts", variable=self.sms_var).pack(anchor=tk.W, padx=5, pady=2)
        
        # Alert rules
        rules_frame = ttk.LabelFrame(self.alerts_tab, text="Alert Rules")
        rules_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(rules_frame, text="Test Alert:").grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(rules_frame, text="Send Test Alert", command=self.send_test_alert).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(rules_frame, text="Alert Cooldown:").grid(row=1, column=0, padx=5, pady=5)
        self.cooldown_var = tk.IntVar(value=5)
        ttk.Spinbox(rules_frame, textvariable=self.cooldown_var, from_=1, to=60, width=5).grid(row=1, column=1, padx=5, pady=5)
    
    def setup_system_tab(self):
        """Setup system controls"""
        # System mode
        mode_frame = ttk.LabelFrame(self.system_tab, text="System Mode")
        mode_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.mode_var = tk.StringVar(value="Active")
        ttk.Radiobutton(mode_frame, text="Active Monitoring", variable=self.mode_var, value="Active").pack(anchor=tk.W, padx=5, pady=2)
        ttk.Radiobutton(mode_frame, text="Silent Monitoring", variable=self.mode_var, value="Silent").pack(anchor=tk.W, padx=5, pady=2)
        ttk.Radiobutton(mode_frame, text="Maintenance Mode", variable=self.mode_var, value="Maintenance").pack(anchor=tk.W, padx=5, pady=2)
        
        # System actions
        action_frame = ttk.LabelFrame(self.system_tab, text="System Actions")
        action_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(action_frame, text="Restart System", command=self.restart_system).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(action_frame, text="Shutdown", command=self.shutdown_system).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(action_frame, text="Export Logs", command=self.export_logs).pack(side=tk.LEFT, padx=5, pady=5)
    
    def refresh_camera_list(self):
        """Refresh list of available cameras"""
        cameras = self.camera_manager.get_available_cameras()
        self.camera_dropdown['values'] = [name for _, name in cameras]
        if cameras:
            self.camera_var.set(cameras[0][1])
    
    def switch_camera(self):
        """Switch to selected camera"""
        camera_name = self.camera_var.get()
        cameras = self.camera_manager.get_available_cameras()
        
        # Find camera ID by name
        for cam_id, name in cameras:
            if name == camera_name:
                self.camera_manager.set_camera(cam_id)
                logger.info(f"Switched to camera: {name}")
                return
        
        logger.error(f"Camera not found: {camera_name}")
    
    def apply_camera_settings(self):
        """Apply camera settings"""
        # Parse resolution
        width, height = map(int, self.res_var.get().split('x'))
        
        # Update config
        config = self.state_manager.get_config()
        config['camera']['resolution']['width'] = width
        config['camera']['resolution']['height'] = height
        config['camera']['fps'] = self.fps_var.get()
        
        self.state_manager.update_config(config)
        logger.info("Camera settings updated")
    
    def apply_detection_settings(self):
        """Apply detection settings"""
        # Update config
        config = self.state_manager.get_config()
        config['detection']['motion']['sensitivity'] = self.motion_sens.get()
        config['detection']['confidence_threshold'] = self.confidence_thresh.get()
        
        # Update toggles
        config['detection']['enable_weapon'] = self.weapon_var.get()
        config['detection']['enable_intruder'] = self.intruder_var.get()
        config['detection']['enable_ppe'] = self.ppe_var.get()
        config['detection']['enable_motion'] = self.motion_var.get()
        
        self.state_manager.update_config(config)
        logger.info("Detection settings updated")
    
    def send_test_alert(self):
        """Send test alert"""
        # Implementation would send a test alert
        logger.info("Test alert sent")
    
    def restart_system(self):
        """Restart the system"""
        # Implementation would restart components
        logger.info("System restart initiated")
    
    def shutdown_system(self):
        """Shutdown the system"""
        # Implementation would safely shut down
        logger.info("System shutdown initiated")
    
    def export_logs(self):
        """Export system logs"""
        # Implementation would export logs
        logger.info("Logs exported")