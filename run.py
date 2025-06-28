import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import joblib

# ---------------------------
# Part 1: Data Preprocessing
# ---------------------------

def extract_frames(video_path, output_dir, fps=1):
    """Extract frames from video at specified FPS"""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    extracted_frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if int(cap.get(cv2.CAP_PROP_FPS)) > 0 and frame_count % int(cap.get(cv2.CAP_PROP_FPS) // fps) == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            extracted_frames.append(frame_path)
            
        frame_count += 1
        
    cap.release()
    return extracted_frames

def load_and_preprocess_data(csv_path, image_dir):
    """Load and preprocess structured and image data"""
    # Load Education Under Attack dataset
    df = pd.read_csv(csv_path)
    
    # Preprocess structured data
    structured_features = ['location_type', 'weapon_type', 'time_of_day']
    numeric_features = ['casualties']
    categorical_features = ['location_type', 'weapon_type', 'time_of_day']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Preprocess image data
    image_paths = []
    labels = []
    structured_data = []
    
    for idx, row in df.iterrows():
    for _, row in df.iterrows():
        img_path = os.path.join(image_dir, f"incident_{row['incident_id']}.jpg")
        if os.path.exists(img_path):
            image_paths.append(img_path)
            labels.append(1 if row['suspicious'] else 0)  # 1=suspicious, 0=normal
            
            # Extract structured features
            structured_row = row[structured_features + numeric_features]
            structured_data.append(structured_row.values)
    # Convert to arrays
    structured_data = np.array(structured_data)
    labels = np.array(labels)
    
    return image_paths, structured_data, labels, preprocessor

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess image for model input"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize
    return img

# ---------------------------
# Part 2: Model Building
# ---------------------------

def create_cnn_model(input_shape):
    """Create CNN model for image processing"""
    inputs = Input(shape=input_shape)
    
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    
    return Model(inputs, x, name='cnn_model')

def create_combined_model(image_shape, num_structured_features):
    """Create combined CNN for images + MLP for structured data"""
    # Image branch
    image_input = Input(shape=image_shape, name='image_input')
    cnn_model = create_cnn_model(image_shape)
    image_output = cnn_model(image_input)
    
    # Structured data branch
    structured_input = Input(shape=(num_structured_features,), name='structured_input')
    x = Dense(32, activation='relu')(structured_input)
    x = Dropout(0.2)(x)
    structured_output = Dense(16, activation='relu')(x)
    
    # Combined model
    combined = concatenate([image_output, structured_output])
    x = Dense(64, activation='relu')(combined)
    x = Dropout(0.3)(x)
    output = Dense(2, activation='softmax')(x)  # 2 classes: suspicious/normal
    
    return Model(inputs=[image_input, structured_input], outputs=output)

# ---------------------------
# Part 3: Training Pipeline
# ---------------------------

def train_model(image_paths, structured_data, labels, preprocessor):
    """Train the combined CNN model"""
    # Preprocess images
    images = np.array([preprocess_image(path) for path in image_paths])
    
    # Preprocess structured data
    structured_data = preprocessor.fit_transform(structured_data)
    
    # Split data
    X_train_img, X_test_img, X_train_struct, X_test_struct, y_train, y_test = train_test_split(
        images, structured_data, labels, test_size=0.2, random_state=42
    )
    
    # Convert labels to categorical
    y_train = to_categorical(y_train, 2)
    y_test = to_categorical(y_test, 2)
    
    # Create model
    model = create_combined_model(images[0].shape, structured_data.shape[1])
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train model
    history = model.fit(
    history = model.fit(
        [X_train_img, X_train_struct],
        y_train,
        epochs=15,
        batch_size=32,
        validation_data=([X_test_img, X_test_struct], y_test)
    )
    
    # Save model and preprocessor
    model.save('surveillance_model.h5')
    joblib.dump(preprocessor, 'preprocessor.joblib')
    
    return model, history
# ---------------------------
# Part 4: Visualization
# ---------------------------

def visualize_data(csv_path):
    """Visualize Education Under Attack dataset"""
    df = pd.read_csv(csv_path)
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Plot incidents over time
    plt.figure(figsize=(12, 6))
    df.set_index('date').resample('M').size().plot()
    plt.title('Education Under Attack Incidents (Monthly)')
    plt.ylabel('Number of Incidents')
    plt.tight_layout()
    plt.savefig('incidents_over_time.png')
    
    # Plot by country
    plt.figure(figsize=(12, 6))
    df['country'].value_counts().head(10).plot(kind='bar')
    plt.title('Top 10 Countries by Incident Count')
    plt.ylabel('Number of Incidents')
    plt.tight_layout()
    plt.savefig('incidents_by_country.png')
    
    # Plot by attack type
    plt.figure(figsize=(12, 6))
    df['attack_type'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Distribution of Attack Types')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('attack_types.png')

# ---------------------------
# Part 5: Tkinter GUI
# ---------------------------

class SurveillanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CCTV Suspicious Activity Detection")
        
        # Load model and preprocessor
        try:
            from tensorflow.keras.models import load_model
            self.model = load_model('surveillance_model.h5')
            self.preprocessor = joblib.load('preprocessor.joblib')
            self.model_loaded = True
        except:
            self.model_loaded = False
            print("Model not found. Please train the model first.")
        
        # Create GUI elements
        self.create_widgets()
        
        # Video capture
        self.cap = None
        self.video_source = 0  # 0 for default camera
        self.is_camera_active = False
        
    def create_widgets(self):
        # Video display
        self.video_label = ttk.Label(self.root)
        self.video_label.pack(pady=10)
        
        # Status display
        self.status_var = tk.StringVar()
        self.status_var.set("Status: Ready")
        self.status_label = ttk.Label(self.root, textvariable=self.status_var, font=('Arial', 12))
        self.status_label.pack(pady=5)
        
        # Alert display
        self.alert_var = tk.StringVar()
        self.alert_var.set("")
        self.alert_label = ttk.Label(
            self.root, 
            textvariable=self.alert_var, 
            font=('Arial', 14, 'bold'),
            foreground='red'
        )
        self.alert_label.pack(pady=5)
        
        # Control buttons
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        self.start_btn = ttk.Button(btn_frame, text="Start Camera", command=self.start_camera)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import joblib

# ---------------------------
# Part 1: Data Preprocessing
# ---------------------------

def extract_frames(video_path, output_dir, fps=1):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    extracted_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if int(cap.get(cv2.CAP_PROP_FPS)) > 0 and frame_count % int(cap.get(cv2.CAP_PROP_FPS) // fps) == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            extracted_frames.append(frame_path)
        frame_count += 1

    cap.release()
    return extracted_frames

def load_and_preprocess_data(csv_path, image_dir):
    df = pd.read_csv(csv_path)

    structured_features = ['location_type', 'weapon_type', 'time_of_day']
    numeric_features = ['casualties']
    categorical_features = ['location_type', 'weapon_type', 'time_of_day']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    image_paths = []
    labels = []
    structured_data = []

    for _, row in df.iterrows():
        img_path = os.path.join(image_dir, f"incident_{row['incident_id']}.jpg")
        if os.path.exists(img_path):
            image_paths.append(img_path)
            labels.append(1 if row['suspicious'] else 0)
            structured_row = row[structured_features + numeric_features]
            structured_data.append(structured_row.values)

    structured_data = np.array(structured_data)
    labels = np.array(labels)
    return image_paths, structured_data, labels, preprocessor

def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return img

# ---------------------------
# Part 2: Model Building
# ---------------------------

def create_cnn_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    return Model(inputs, x, name='cnn_model')

def create_combined_model(image_shape, num_structured_features):
    image_input = Input(shape=image_shape, name='image_input')
    cnn_model = create_cnn_model(image_shape)
    image_output = cnn_model(image_input)

    structured_input = Input(shape=(num_structured_features,), name='structured_input')
    x = Dense(32, activation='relu')(structured_input)
    x = Dropout(0.2)(x)
    structured_output = Dense(16, activation='relu')(x)

    combined = concatenate([image_output, structured_output])
    x = Dense(64, activation='relu')(combined)
    x = Dropout(0.3)(x)
    output = Dense(2, activation='softmax')(x)

    return Model(inputs=[image_input, structured_input], outputs=output)

# ---------------------------
# Part 3: Training Pipeline
# ---------------------------

def train_model(image_paths, structured_data, labels, preprocessor):
    images = np.array([preprocess_image(path) for path in image_paths])
    structured_data = preprocessor.fit_transform(structured_data)

    X_train_img, X_test_img, X_train_struct, X_test_struct, y_train, y_test = train_test_split(
        images, structured_data, labels, test_size=0.2, random_state=42
    )

    y_train = to_categorical(y_train, 2)
    y_test = to_categorical(y_test, 2)

    model = create_combined_model(images[0].shape, structured_data.shape[1])
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        [X_train_img, X_train_struct],
        y_train,
        epochs=15,
        batch_size=32,
        validation_data=([X_test_img, X_test_struct], y_test)
    )

    model.save('surveillance_model.h5')
    joblib.dump(preprocessor, 'preprocessor.joblib')
    return model, history

        self.stop_btn = ttk.Button(btn_frame, text="Stop Camera", command=self.stop_camera, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.file_btn = ttk.Button(btn_frame, text="Open Video File", command=self.open_file)
        self.file_btn.pack(side=tk.LEFT, padx=5)
        
        # Structured data simulation (in real system, get from sensors/database)
        self.structured_data = np.array([[2, 3, 1, 18]])  # Example: [location_type, weapon_type, time_of_day, casualties]
        
    def start_camera(self):
        """Start camera capture"""
        if not self.model_loaded:
            self.status_var.set("Error: Model not loaded")
            return
            
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            self.status_var.set("Error: Camera not accessible")
            return
            
        self.is_camera_active = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set("Status: Live camera active")
        self.process_video()
        
    def stop_camera(self):
        """Stop camera capture"""
        self.is_camera_active = False
        if self.cap:
            self.cap.release()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("Status: Camera stopped")
        
    def open_file(self):
        """Open video file for processing"""
        if not self.model_loaded:
            self.status_var.set("Error: Model not loaded")
            return
            
        file_path = filedialog.askopenfilename(
            filetypes=[("Video Files", "*.mp4 *.avi *.mov"), ("All Files", "*.*")]
        )
        if not file_path:
            return
            
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            self.status_var.set("Error: Could not open video file")
            return
            
        self.is_camera_active = True
        self.status_var.set(f"Status: Processing {os.path.basename(file_path)}")
        self.process_video()
        
    def process_video(self):
        """Process video frames and run detection"""
        if not self.cap or not self.cap.isOpened():
            return
            
        ret, frame = self.cap.read()
        if ret:
            # Preprocess frame
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            display_frame = cv2.resize(display_frame, (640, 480))
            
            # Run detection every 5 frames
            if hasattr(self, 'frame_count'):
                self.frame_count += 1
            else:
                self.frame_count = 0
                
            if self.frame_count % 5 == 0:
                # Preprocess for model
                model_frame = cv2.resize(frame, (224, 224))
                model_frame = model_frame / 255.0
                
                # Preprocess structured data
                processed_struct = self.preprocessor.transform(self.structured_data)
                
                # Make prediction
                prediction = self.model.predict([np.array([model_frame]), processed_struct])
                suspicious_prob = prediction[0][1]  # Probability of being suspicious
                
                # Update status
                if suspicious_prob > 0.7:
                    self.alert_var.set("ALERT: Suspicious Activity Detected!")
                    # Draw red border on frame
                    display_frame = cv2.copyMakeBorder(
                        display_frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 0, 0)
                    )
                else:
                    self.alert_var.set("")
                
                self.status_var.set(f"Status: Monitoring - Suspicious: {suspicious_prob:.2f}")
            
            # Display frame
            img = Image.fromarray(display_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            
        if self.is_camera_active:
            self.root.after(10, self.process_video)
        else:
            if self.cap:
                self.cap.release()

# ---------------------------
# Main Execution
# ---------------------------

if __name__ == "__main__":
    # Step 1: Data Preparation (Run this part in Jupyter first)
    # --------------------------------------------------------
    # Uncomment to run data processing and training
    
    # # Load data
    # image_paths, structured_data, labels, preprocessor = load_and_preprocess_data(
    #     'education_attacks.csv', 
    #     'dataset/images'
    # )
    
    # # Visualize dataset
    # visualize_data('education_attacks.csv')
    
    # # Train model
    # model, history = train_model(image_paths, structured_data, labels, preprocessor)
    
    # Step 2: Run the GUI application
    # -------------------------------
    root = tk.Tk()
    app = SurveillanceApp(root)
    root.geometry("800x700")
    root.mainloop()