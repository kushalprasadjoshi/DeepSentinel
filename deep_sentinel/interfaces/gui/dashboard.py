import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from deep_sentinel.utils import logging_utils

logger = logging_utils.setup_module_logger(__name__)

class ThreatDashboard(ttk.Frame):
    """Dashboard for visualizing security threats and system metrics
    
    Attributes:
        state_manager: SystemState instance
        threat_chart: Matplotlib threat distribution chart
        metrics_frame: System metrics display
        alert_list: Listbox for alert history
    """
    
    def __init__(self, parent, state_manager):
        """
        Initialize threat dashboard
        
        Args:
            parent: Parent widget
            state_manager: SystemState instance
        """
        super().__init__(parent)
        self.state_manager = state_manager
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=0, column=0, sticky="nsew")
        
        # Create tabs
        self.threat_tab = ttk.Frame(self.notebook)
        self.metrics_tab = ttk.Frame(self.notebook)
        self.alerts_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.threat_tab, text="Threat Analysis")
        self.notebook.add(self.metrics_tab, text="System Metrics")
        self.notebook.add(self.alerts_tab, text="Alert History")
        
        # Setup each tab
        self.setup_threat_tab()
        self.setup_metrics_tab()
        self.setup_alerts_tab()
        
        # Start update timer
        self.update_dashboard()
    
    def setup_threat_tab(self):
        """Setup threat analysis tab"""
        # Threat distribution chart
        fig = Figure(figsize=(6, 4), dpi=100)
        self.threat_chart = fig.add_subplot(111)
        self.threat_chart.set_title("Threat Distribution")
        self.threat_chart.set_xlabel("Threat Type")
        self.threat_chart.set_ylabel("Count")
        
        self.chart_canvas = FigureCanvasTkAgg(fig, self.threat_tab)
        self.chart_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Threat statistics panel
        stats_frame = ttk.LabelFrame(self.threat_tab, text="Threat Statistics")
        stats_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        self.threat_stats = ttk.Label(stats_frame, text="Total Threats: 0 | Today: 0")
        self.threat_stats.pack(pady=5)
        
        self.top_threat = ttk.Label(stats_frame, text="Most Common: None")
        self.top_threat.pack(pady=5)
    
    def setup_metrics_tab(self):
        """Setup system metrics tab"""
        # System metrics display
        metrics_frame = ttk.LabelFrame(self.metrics_tab, text="System Performance")
        metrics_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.fps_label = ttk.Label(metrics_frame, text="FPS: 0.0")
        self.fps_label.pack(pady=5)
        
        self.cpu_label = ttk.Label(metrics_frame, text="CPU Usage: 0%")
        self.cpu_label.pack(pady=5)
        
        self.mem_label = ttk.Label(metrics_frame, text="Memory Usage: 0 MB")
        self.mem_label.pack(pady=5)
        
        # Camera status panel
        camera_frame = ttk.LabelFrame(self.metrics_tab, text="Camera Status")
        camera_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.camera_status = ttk.Label(camera_frame, text="Camera 0: Active")
        self.camera_status.pack(pady=5)
    
    def setup_alerts_tab(self):
        """Setup alert history tab"""
        # Alert history list
        alert_frame = ttk.Frame(self.alerts_tab)
        alert_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scrollbar = ttk.Scrollbar(alert_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.alert_list = tk.Listbox(
            alert_frame, 
            yscrollcommand=scrollbar.set,
            height=15,
            font=("Arial", 10)
        )
        self.alert_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.alert_list.yview)
        
        # Alert details panel
        detail_frame = ttk.LabelFrame(self.alerts_tab, text="Alert Details")
        detail_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.alert_detail = ttk.Label(detail_frame, text="Select an alert to view details")
        self.alert_detail.pack(pady=5)
    
    def update_dashboard(self):
        """Update dashboard components"""
        # Update threat chart
        self.update_threat_chart()
        
        # Update system metrics
        self.update_metrics()
        
        # Update alert history
        self.update_alerts()
        
        # Schedule next update
        self.after(5000, self.update_dashboard)
    
    def update_threat_chart(self):
        """Update threat distribution chart"""
        # Get threat data
        threats = self.state_manager.threat_events
        threat_counts = {}
        
        for threat in threats:
            t_type = threat['type']
            threat_counts[t_type] = threat_counts.get(t_type, 0) + 1
        
        # Clear previous chart
        self.threat_chart.clear()
        
        # Create new chart
        if threat_counts:
            types = list(threat_counts.keys())
            counts = [threat_counts[t] for t in types]
            
            self.threat_chart.bar(types, counts, color='red')
            self.threat_chart.set_title(f"Threat Distribution ({sum(counts)} total)")
            self.threat_chart.set_xlabel("Threat Type")
            self.threat_chart.set_ylabel("Count")
            
            # Update statistics
            total = sum(counts)
            today = sum(1 for t in threats if time.time() - t['timestamp'] < 86400)
            most_common = max(threat_counts, key=threat_counts.get)
            
            self.threat_stats.config(text=f"Total Threats: {total} | Today: {today}")
            self.top_threat.config(text=f"Most Common: {most_common} ({threat_counts[most_common]})")
        else:
            self.threat_chart.set_title("No Threats Detected")
            self.threat_stats.config(text="Total Threats: 0 | Today: 0")
            self.top_threat.config(text="Most Common: None")
        
        self.chart_canvas.draw()
    
    def update_metrics(self):
        """Update system metrics display"""
        status = self.state_manager.system_status
        
        self.fps_label.config(text=f"FPS: {status.get('fps', 0.0):.1f}")
        self.cpu_label.config(text=f"CPU Usage: {status.get('cpu_usage', 0):.1f}%")
        self.mem_label.config(text=f"Memory Usage: {status.get('memory_usage', 0):.1f} MB")
        
        # Update camera status
        cameras = self.state_manager.camera_states
        if cameras:
            cam_id = list(cameras.keys())[0]
            state = "Active" if cameras[cam_id].get('active', False) else "Inactive"
            self.camera_status.config(text=f"Camera {cam_id}: {state}")
    
    def update_alerts(self):
        """Update alert history list"""
        alerts = self.state_manager.alert_history
        self.alert_list.delete(0, tk.END)
        
        