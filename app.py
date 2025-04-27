from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import os, cv2, time, datetime
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

from functools import wraps
from flask import make_response

def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    return no_cache


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # needed for sessions

# Admin login constants
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'pass123'

#Route for home page
@app.route('/')
def home():
    return render_template("index.html")

# Route: Admin login
@app.route('/login', methods=['POST'])
def login():
    admin_id = request.form['admin_id']
    password = request.form['password']
    if admin_id == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        session['admin_logged_in'] = True
        return redirect(url_for('dashboard'))
    else:
        return render_template("index.html", error="❌ Invalid ID or Password")

# Route: admin Dashboard 
@app.route('/dashboard')
@nocache
def dashboard():
    if not session.get('admin_logged_in'):
        return redirect(url_for('home'))

    # Calculate total users
    faces_dir = 'faces_dataset'
    if not os.path.exists(faces_dir):
        os.makedirs(faces_dir)
    total_users = len(os.listdir(faces_dir))

    return render_template("admin_dashboard.html", total_users=total_users)

# Route: Admin Logout
@app.route('/logout')
def logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('home'))



# Route: Register new user
@app.route('/register_user', methods=['GET', 'POST'])
def register_user():
    if request.method == 'POST':
        user_id = request.form['user_id']
        username = request.form['username']

        if user_id == '' or username == '':
            flash('Please enter both ID and Name.')
            return redirect(url_for('register_user'))

        # Save both in session to pass for capturing
        session['new_user_id'] = user_id
        session['new_username'] = username

        return redirect(url_for('capture_faces'))

    return render_template('register_user.html')


# Route: Capture Faces
@app.route('/capture_faces')
def capture_faces():
    user_id = session.get('new_user_id')
    username = session.get('new_username')

    if not user_id or not username:
        flash('User info missing. Please register again.')
        return redirect(url_for('register_user'))

    # Directory for faces
    faces_dir = 'faces_dataset'
    if not os.path.exists(faces_dir):
        os.makedirs(faces_dir)

    # Create a new folder for this user
    user_folder = os.path.join(faces_dir, f"{user_id}_{username}")
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    # Start capturing from webcam
    cap = cv2.VideoCapture(0)
    count = 0

    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))  # Resize face to 200x200
            cv2.imwrite(os.path.join(user_folder, f"{count}.jpg"), face)

            # Show rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            time.sleep(0.3)

        cv2.imshow('Capturing Faces - Press Q to Exit', frame)

        if cv2.waitKey(1) == ord('q') or count >= 40:
            break

    cap.release()
    cv2.destroyAllWindows()

    flash('✅ Face capture successful! Now you can Train the Model.', 'success')
    return redirect(url_for('dashboard'))



# Route: Train model
@app.route('/train_model')
def train_model():
    faces_dir = 'faces_dataset'
    model_path = 'src_code/face_recognition_model.h5'
    timestamp_path = 'src_code/last_trained.txt'

    # Check if faces directory exists and is not empty
    if not os.path.exists(faces_dir) or len(os.listdir(faces_dir)) == 0:
        flash('⚠️ No users registered. Please register users first!', 'danger')
        return redirect(url_for('dashboard'))

    # Check if model exists
    model_exists = os.path.exists(model_path)

    # Check if last trained timestamp exists
    if model_exists and os.path.exists(timestamp_path):
        with open(timestamp_path, 'r') as f:
            last_trained_time = float(f.read().strip())
        
        # Check if any user folder was modified after last training
        new_data_found = False
        for folder in os.listdir(faces_dir):
            user_folder = os.path.join(faces_dir, folder)
            if os.path.isdir(user_folder):
                if os.path.getmtime(user_folder) > last_trained_time:
                    new_data_found = True
                    break
        
        if new_data_found:
            flash('⚠️ New users are detected! Training the model.', 'warning')
    else:
        flash('⚠️ Training the model...', 'warning')

    images = []
    labels = []

    # Read user's images
    for folder in os.listdir(faces_dir):
        user_folder = os.path.join(faces_dir, folder)
        if os.path.isdir(user_folder):
            for img_file in os.listdir(user_folder):
                img_path = os.path.join(user_folder, img_file)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (100, 100))
                images.append(img)
                labels.append(folder)

    if len(images) == 0:
        flash('⚠️ No face images found. Please capture faces.', 'danger')
        return redirect(url_for('dashboard'))

    images = np.array(images).reshape(-1, 100, 100, 1) / 255.0
    labels = np.array(labels)

    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    labels_encoded = to_categorical(labels_encoded)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

    # Build CNN
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(100,100,1)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(labels_encoded.shape[1], activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    # Save Model
    model.save(model_path)
    np.save('src_code/label_classes.npy', le.classes_)

    # Save timestamp
    with open(timestamp_path, 'w') as f:
        f.write(str(time.time()))

    flash('✅ Model trained successfully!', 'success')
    return redirect(url_for('dashboard'))



if __name__ == '__main__':
    app.run(debug=True)
