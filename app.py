from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import os, cv2, time, datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from attendance_routes import attendance_bp
from manageUser_routes import manage_users

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
app.register_blueprint(attendance_bp)
app.register_blueprint(manage_users)
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


@app.route('/capture_faces')
def capture_faces():
    user_id = session.get('new_user_id')
    username = session.get('new_username')
    if not user_id or not username:
        flash('User info missing. Please register again.')
        return redirect(url_for('register_user'))

    faces_dir = 'faces_dataset'
    user_folder = os.path.join(faces_dir, f"{user_id}_{username}")
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    count = 0
    max_images = 120
    video_duration = 10  # seconds
    start_time = time.time()


    while count < max_images and (time.time() - start_time) < video_duration:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, "Move your head slowly for 10 seconds", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            if count < max_images:
                face = frame[y:y+h, x:x+w]
                face = cv2.resize(face, (100, 100))  # Consistent size with training
                img_name = f"{count+1}.jpg"
                cv2.imwrite(os.path.join(user_folder, img_name), face)
                count += 1
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Capturing Faces - Recording', frame)
        if cv2.waitKey(1) == ord('q'):
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
    label_classes_path = 'src_code/label_classes.npy'

    # Check if faces directory exists and is not empty
    if not os.path.exists(faces_dir) or len(os.listdir(faces_dir)) == 0:
        flash('⚠️ No users registered. Please register users first!', 'danger')
        return redirect(url_for('dashboard'))

    # Remove old model and label files to ensure clean training
    if os.path.exists(model_path):
        os.remove(model_path)
    if os.path.exists(label_classes_path):
        os.remove(label_classes_path)

    images = []
    labels = []

    # Read user's images
    for folder in os.listdir(faces_dir):
        user_folder = os.path.join(faces_dir, folder)
        if os.path.isdir(user_folder):
            for img_file in os.listdir(user_folder):
                img_path = os.path.join(user_folder, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                img = clahe.apply(img)
                img = cv2.GaussianBlur(img, (5, 5), 0)
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

    # Save the current label classes
    np.save(label_classes_path, le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

    # Inside /train_model route, after splitting the data
    datagen = ImageDataGenerator(
        rotation_range=20,        
        width_shift_range=0.2,    
        height_shift_range=0.2,   
        shear_range=0.2,          
        zoom_range=0.2,          
        horizontal_flip=True,    
        fill_mode='nearest'      
    )

    # Building CNN model
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(100,100,1)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Conv2D(256, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(labels_encoded.shape[1], activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

  # Train the model with augmented data
    model.fit(datagen.flow(X_train, y_train, batch_size=32),
              epochs=30,
              validation_data=(X_test, y_test),
              callbacks=[early_stopping])

    # Save Model
    model.save(model_path)

    # Save timestamp
    with open(timestamp_path, 'w') as f:
        f.write(str(time.time()))

    flash('✅ Model trained successfully!', 'success')
    return redirect(url_for('dashboard'))



if __name__ == '__main__':
    app.run(debug=True)





