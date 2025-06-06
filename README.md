# Face Recognition Attendance System

## Overview
A Python-based attendance system that uses deep learning for face recognition to automatically mark attendance. The system employs a CNN model trained on face datasets to recognize individuals in real-time through a webcam. It captures images, processes them through the trained model, and records attendance based on successful face recognition.

## Features
- Deep learning-based face recognition using CNN
- Real-time face detection and recognition
- Automatic attendance marking
- Support for multiple faces in a single frame
- Export attendance records to CSV
- User-friendly interface
- Model training capability for new faces

## How It Works
1. **Model Training**:
   - System captures multiple images of a person's face
   - Images are preprocessed and augmented
   - CNN model is trained on the collected face data
   - Model weights are saved for future recognition

2. **Face Detection and Recognition**:
   - Real-time face detection using OpenCV
   - Detected faces are processed through the trained CNN model
   - Model outputs confidence scores for each recognized face
   - Faces with confidence above threshold are marked as recognized

3. **Attendance Marking**:
   - When a face is recognized with high confidence:
     - Records the timestamp
     - Updates the attendance record
     - Provides visual feedback
   - Attendance data is saved in CSV format

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Webcam
- Required Python packages (listed in requirements.txt)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/VarshaSingh-343/face_recognition_attendance_system.git
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Linux/Mac
   source venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
1. Run the main application:
   ```bash
   python main.py
   ```

2. Train the model with new faces:
   - Use the "Add New Face" feature
   - Capture multiple images of the person
   - System will train the model on new face data
   - Wait for training completion

3. Start attendance marking:
   - The system will automatically detect and recognize faces
   - Attendance will be marked in real-time
   - Records can be exported to CSV format

## Contributing
Feel free to submit issues and enhancement requests!

## License
This project is licensed under the MIT License - see the LICENSE file for details. 