from flask import Blueprint, render_template, redirect, url_for, flash, request, session, jsonify
import os
import shutil
from datetime import datetime
import pandas as pd

manage_users = Blueprint('manage_users', __name__)

@manage_users.route('/manage-users')
def view_users():
    """View all registered users"""
    if not session.get('admin_logged_in'):
        return redirect(url_for('home'))
    
    # Get list of users from faces_dataset directory
    faces_dir = 'faces_dataset'
    if not os.path.exists(faces_dir):
        os.makedirs(faces_dir)
        return render_template('manage_users.html', users=[])
    
    users = []
    for folder in os.listdir(faces_dir):
        if os.path.isdir(os.path.join(faces_dir, folder)):
            if '_' in folder:
                user_id, username = folder.split('_', 1)
            else:
                user_id = folder
                username = "Unknown"
            
            # Get registration date from folder creation time
            folder_path = os.path.join(faces_dir, folder)
            reg_date = datetime.fromtimestamp(os.path.getctime(folder_path)).strftime('%Y-%m-%d %H:%M:%S')
            
            users.append({
                'id': user_id,
                'username': username,
                'registration_date': reg_date,
                'folder_name': folder
            })
    
    return render_template('manage_users.html', users=users)

@manage_users.route('/manage-users/attendance/<string:user_id>')
def get_user_attendance(user_id):
    """Get attendance records for a specific user"""
    if not session.get('admin_logged_in'):
        return jsonify([])
    
    attendance_file = 'attendance/attendance_output.csv'
    if not os.path.exists(attendance_file):
        print(f"Attendance file not found at: {attendance_file}")
        return jsonify([])
    
    try:
        # Read the CSV file
        df = pd.read_csv(attendance_file)
        print(f"Total records in file: {len(df)}")
        
        # Convert user_id to string for comparison
        user_id = str(user_id)
        
        # Filter records for the specific user
        user_records = df[df['User ID'].astype(str) == user_id]
        print(f"Records found for user {user_id}: {len(user_records)}")
        
        # Convert to list of dictionaries
        records = []
        for _, row in user_records.iterrows():
            records.append({
                'date': str(row['Date']),
                'time': str(row['Time'])
            })
        
        print(f"Returning {len(records)} records")
        return jsonify(records)
    except Exception as e:
        print(f"Error reading attendance data: {str(e)}")
        return jsonify([])

@manage_users.route('/manage-users/delete/<string:user_id>', methods=['POST'])
def delete_user(user_id):
    """Delete a user and their associated data"""
    if not session.get('admin_logged_in'):
        return redirect(url_for('home'))
    
    faces_dir = 'faces_dataset'
    user_found = False
    
    # Find and delete the user's folder
    for folder in os.listdir(faces_dir):
        if folder.startswith(f"{user_id}_") or folder == user_id:
            folder_path = os.path.join(faces_dir, folder)
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
                user_found = True
                break

    if user_found:
        flash(f'User {user_id} has been successfully deleted. Please train the model again for accurate face recognition.', 'success')
        # Redirect to train model page
        return redirect(url_for('train_model'))
    else:
        flash(f'User {user_id} not found.', 'danger')
    
    return redirect(url_for('manage_users.view_users')) 