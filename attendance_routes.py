from flask import Blueprint, request, render_template, redirect, url_for, flash, session, send_from_directory
import cv2
import time
import numpy as np
import os
import csv
import datetime
import pandas as pd
from tensorflow.keras.models import load_model

attendance_bp = Blueprint('attendance', __name__)

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to check if a user folder exists
def is_valid_user(user_folder_name):
    faces_dir = 'faces_dataset'
    if '_' in user_folder_name:
        user_id, username = user_folder_name.split('_', 1)
        user_folder = os.path.join(faces_dir, f"{user_id}_{username}")
    else:
        user_folder = os.path.join(faces_dir, user_folder_name)
    return os.path.exists(user_folder) and os.path.isdir(user_folder)

# Taking attendance route
@attendance_bp.route('/take_attendance')
def take_attendance():
    if not session.get('admin_logged_in'):
        return redirect(url_for('home'))

    # Load model and labels with error handling
    model_path = 'src_code/face_recognition_model.h5'
    label_classes_path = 'src_code/label_classes.npy'
    try:
        model = load_model(model_path)
        labels = np.load(label_classes_path)
    except Exception as e:
        flash(f'⚠️ Failed to load model or labels: {str(e)}', 'danger')
        return redirect(url_for('dashboard'))

    # Start webcam with error handling
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        flash('⚠️ Failed to access webcam.', 'danger')
        return redirect(url_for('dashboard'))

    user_tracking = {}  # Dictionary to track confidence scores and frames for each user
    min_frames_required = 5  # Reduced for faster verification
    max_tracking_frames = 20  # Increased for more robust tracking
    final_recognized_users = set()  # Set to store final recognized users
    attendance_status = {}  # Dictionary to track attendance status
    cooldown_period = 20  # Reduced cooldown period
    attendance_recorded = False  # Track if attendance has been recorded
    frame_count = 0  # For frame skipping
    skip_frames = 2  # Process every 2nd frame

    while True:
        ret, frame = cap.read()
        if not ret:
            flash('⚠️ Failed to capture frame from webcam.', 'warning')
            break

        frame_count += 1
        if frame_count % skip_frames != 0:
            cv2.imshow('Attendance', frame)
            time.sleep(0.2)
            if cv2.waitKey(1) == ord('q'):
                break
            continue

        # Resize frame for faster face detection
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Scale coordinates back to original frame size
            x, y, w, h = [int(v * 2) for v in (x, y, w, h)]
            face = frame[y:y+h, x:x+w]
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            face_gray = clahe.apply(face_gray)
            face_gray = cv2.GaussianBlur(face_gray, (5, 5), 0)
            face_resized = cv2.resize(face_gray, (100, 100))
            face_input = face_resized.reshape(1, 100, 100, 1) / 255.0

            prediction = model.predict(face_input)
            class_index = np.argmax(prediction)
            confidence = prediction[0][class_index]

            if confidence > 0.85:  # Lowered threshold
                user_folder_name = labels[class_index]

                # Check if this is a valid user
                if not is_valid_user(user_folder_name):
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, 'Invalid User', (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    continue

                # Initialize tracking for new user
                if user_folder_name not in user_tracking:
                    user_tracking[user_folder_name] = {
                        'confidences': [],
                        'frames': 0,
                        'last_seen': 0,
                        'verified': False,
                        'cooldown': 0
                    }
                    attendance_status[user_folder_name] = False

                # Update tracking data
                tracking = user_tracking[user_folder_name]

                # Skip if in cooldown period
                if tracking['cooldown'] > 0:
                    tracking['cooldown'] -= 1
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, f'Cooldown: {tracking["cooldown"]}', (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    continue

                tracking['confidences'].append(confidence)
                tracking['frames'] += 1
                tracking['last_seen'] = 0

                # Keep only recent confidences
                if len(tracking['confidences']) > max_tracking_frames:
                    tracking['confidences'] = tracking['confidences'][-max_tracking_frames:]

                # Check if we have enough frames and consistent high confidence
                if tracking['frames'] >= min_frames_required:
                    recent_confidences = tracking['confidences'][-min_frames_required:]
                    avg_confidence = np.mean(recent_confidences)
                    min_confidence = np.min(recent_confidences)

                    # Adjusted thresholds for robustness
                    if avg_confidence > 0.85 and min_confidence > 0.80:
                        if not attendance_status[user_folder_name] and not attendance_recorded:
                            tracking['verified'] = True
                            final_recognized_users.add(user_folder_name)
                            attendance_status[user_folder_name] = True
                            attendance_recorded = True
                            tracking['cooldown'] = cooldown_period
                            # Draw green rectangle for verified recognition
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(frame, f'{user_folder_name} (Attendance Recorded)', (x, y-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                        else:
                            # Draw blue rectangle for already recorded
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                            cv2.putText(frame, f'{user_folder_name} (Already Recorded)', (x, y-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    else:
                        # Draw yellow rectangle for potential match
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                        cv2.putText(frame, f'Verifying... ({tracking["frames"]}/{min_frames_required})',
                                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                else:
                    # Draw yellow rectangle while collecting frames
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                    cv2.putText(frame, f'Verifying... ({tracking["frames"]}/{min_frames_required})',
                                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            else:
                # Unknown face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, f'Unknown ({confidence:.2f})', (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Increment last_seen counter for all users
        for user in user_tracking:
            user_tracking[user]['last_seen'] += 1
            # Reset tracking if user hasn't been seen for a while
            if user_tracking[user]['last_seen'] > max_tracking_frames:
                user_tracking[user] = {
                    'confidences': [],
                    'frames': 0,
                    'last_seen': 0,
                    'verified': False,
                    'cooldown': 0
                }

        # Display attendance status
        if attendance_recorded:
            cv2.putText(frame, "Attendance Recorded - Press Q to Exit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Press Q to Stop", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Attendance', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save attendance only for verified users
    if not os.path.exists('attendance'):
        os.makedirs('attendance')

    today_date = datetime.date.today().strftime("%d-%m-%Y")
    current_time = datetime.datetime.now().strftime("%H:%M:%S")

    daily_filename = f'attendance/attendance_{today_date}.csv'
    master_filename = 'attendance/attendance_output.csv'

    # Ensure daily file and master file headers
    for filename in [daily_filename, master_filename]:
        if not os.path.exists(filename):
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['User ID', 'Name', 'Date', 'Time'])

    # Write only verified users
    for user_folder in final_recognized_users:
        if '_' in user_folder:
            user_id, name = user_folder.split('_', 1)
        else:
            user_id = user_folder
            name = "Unknown"

        row = [user_id, name, today_date, current_time]

        # Write into today's attendance file
        with open(daily_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        # Write into master file
        with open(master_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    flash('✅ Attendance captured successfully!', 'success')
    return redirect(url_for('dashboard'))

# Route to View Attendance Records
@attendance_bp.route('/view_attendance', methods=['GET', 'POST'])
def view_attendance():
    master_file_path = os.path.join('attendance', 'attendance_output.csv')

    # Check if master file exists
    if not os.path.exists(master_file_path):
        flash('⚠️ No attendance records found yet.', 'warning')
        return redirect(url_for('dashboard'))

    try:
        # Read the CSV file with error handling
        df = pd.read_csv(master_file_path)
        
        # Drop any rows with all NaN values
        df = df.dropna(how='all')
        
        # Fill NaN values in specific columns with appropriate values
        df['User ID'] = df['User ID'].fillna('Unknown')
        df['Name'] = df['Name'].fillna('Unknown')
        df['Date'] = df['Date'].fillna('Unknown')
        df['Time'] = df['Time'].fillna('Unknown')
        
        # Ensure the 'Date' column has no extra spaces or newlines
        df['Date'] = df['Date'].str.strip()
        
        # Filter by selected date if any (via GET parameter)
        selected_date = request.args.get('date')
        if selected_date:
            # Convert selected_date from YYYY-MM-DD to DD-MM-YYYY format
            try:
                selected_date_obj = datetime.datetime.strptime(selected_date, '%Y-%m-%d')
                selected_date_formatted = selected_date_obj.strftime('%d-%m-%Y')
                df = df[df['Date'] == selected_date_formatted]
            except ValueError:
                flash('⚠️ Invalid date format.', 'warning')
                return redirect(url_for('attendance.view_attendance'))
        
        # Convert the DataFrame to a list of dictionaries
        attendance_records = df.to_dict(orient='records')
        
        # Clean the records to ensure no NaN values
        cleaned_records = []
        for record in attendance_records:
            cleaned_record = {
                'User ID': str(record.get('User ID', 'Unknown')),
                'Name': str(record.get('Name', 'Unknown')),
                'Date': str(record.get('Date', 'Unknown')),
                'Time': str(record.get('Time', 'Unknown'))
            }
            cleaned_records.append(cleaned_record)

        return render_template('view_attendance.html', attendance_records=cleaned_records)
    
    except Exception as e:
        flash(f'⚠️ Error reading attendance records: {str(e)}', 'danger')
        return redirect(url_for('dashboard'))

# Route to Export Attendance Records as CSV
@attendance_bp.route('/export_attendance', methods=['POST'])
def export_attendance():
    master_file_path = os.path.join('attendance', 'attendance_output.csv')

    # Check if master file exists
    if not os.path.exists(master_file_path):
        flash('⚠️ No attendance records found yet.', 'warning')
        return redirect(url_for('dashboard'))

    # Read the CSV file
    df = pd.read_csv(master_file_path)

    # Ensure the 'Date' column has no extra spaces or newlines
    df['Date'] = df['Date'].str.strip()

    # Get the date filter from form or args
    selected_date = request.form.get('date') or request.args.get('date')
    if selected_date:
        # Convert selected_date from YYYY-MM-DD to DD-MM-YYYY format
        try:
            selected_date_obj = datetime.datetime.strptime(selected_date, '%Y-%m-%d')
            selected_date_formatted = selected_date_obj.strftime('%d-%m-%Y')
            df = df[df['Date'] == selected_date_formatted]
        except ValueError:
            flash('⚠️ Invalid date format.', 'warning')
            return redirect(url_for('attendance.view_attendance'))

    # Set the file name for the export
    export_filename = f'attendance_export_{datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}.csv'

    # Export the filtered data to CSV
    export_path = os.path.join('attendance', export_filename)
    df.to_csv(export_path, index=False)

    # Send the file to the user for download
    return redirect(url_for('attendance.download_file', filename=export_filename))

# Route to download exported file
@attendance_bp.route('/download_file/<filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory('attendance', filename, as_attachment=True)