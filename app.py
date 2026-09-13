from flask import Flask, render_template, request, redirect, url_for, Response, send_file, flash, session, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from concurrent.futures import ThreadPoolExecutor
import cv2
import io
import hmac
import os
import numpy as np
import face_recognition
from werkzeug.security import check_password_hash
from database import Database
from recognition import get_face_embedding
from camera import get_frame

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-me')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 31536000

FRAME_WIDTH = 640
FRAME_HEIGHT = 360
CAMERA_FPS = 15
RECOGNITION_SCALE = 0.25
PROCESS_EVERY_N_FRAMES = 2
HOG_FALLBACK_EVERY_N_FRAMES = 20
JPEG_QUALITY = 70
MATCH_THRESHOLD = 0.6
MIN_FACE_SIZE = (24, 24)
ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'admin')
ADMIN_PASSWORD_HASH = os.environ.get('ADMIN_PASSWORD_HASH')

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

db = Database()
students = []  # Will be loaded on startup
known_face_ids = []
known_face_names = []
known_face_embeddings = np.empty((0, 128))
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
attendance_executor = ThreadPoolExecutor(max_workers=1)

class User(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

def load_students():
    """Load students from database."""
    global students, known_face_ids, known_face_names, known_face_embeddings
    students = db.get_students()
    known_face_ids = [student['id'] for student in students]
    known_face_names = [student['name'] for student in students]
    if students:
        known_face_embeddings = np.asarray([student['embedding'] for student in students])
    else:
        known_face_embeddings = np.empty((0, 128))

def valid_admin_login(username, password):
    if not hmac.compare_digest(username, ADMIN_USERNAME):
        return False
    if ADMIN_PASSWORD_HASH:
        return check_password_hash(ADMIN_PASSWORD_HASH, password)
    return hmac.compare_digest(password, ADMIN_PASSWORD)


def detect_faces_fast(small_frame, frame_count):
    """Return face locations in face_recognition's (top, right, bottom, left) format."""
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=MIN_FACE_SIZE
    )
    if len(faces):
        return [(y, x + w, y + h, x) for (x, y, w, h) in faces]

    if frame_count % HOG_FALLBACK_EVERY_N_FRAMES == 0:
        rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        return face_recognition.face_locations(rgb, model='hog')

    return []

def mark_attendance_async(student_id):
    """Mark attendance outside the camera loop so video frames do not pause on DB writes."""
    attendance_executor.submit(db.mark_attendance, student_id)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if valid_admin_login(username, password):
            user = User(1)
            login_user(user)
            load_students()  # Load students after login
            return redirect(url_for('dashboard'))
        flash('Invalid username or password.', 'error')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def dashboard():
    total_students = db.get_total_students()
    today_attendance = db.get_attendance_today()
    today_count = db.get_attendance_count_today()
    all_attendance = db.get_all_attendance(limit=200)
    student_summary = db.get_student_summaries()
    last_capture = session.get('last_capture')
    return render_template(
        'index.html',
        total=total_students,
        today=today_attendance,
        today_count=today_count,
        student_summary=student_summary,
        last_capture=last_capture,
        all=all_attendance
    )

@app.route('/dashboard_data')
@login_required
def dashboard_data():
    """Return fresh dashboard data without reloading the whole page."""
    return jsonify({
        'total': db.get_total_students(),
        'today_count': db.get_attendance_count_today(),
        'student_summary': db.get_student_summaries(),
        'attendance': db.get_all_attendance(limit=200)
    })

@app.route('/register', methods=['GET', 'POST'])
@login_required
def register():
    if request.method == 'POST':
        name = request.form['name'].strip()
        department = request.form['department'].strip()
        if not name or not department:
            flash('Student name and department are required.', 'error')
            return render_template('register.html')

        frame = get_frame()
        if frame is not None:
            emb = get_face_embedding(frame)
            if emb is not None:
                db.add_student(name, department, emb)
                load_students()  # Reload students
                session['last_capture'] = name
                flash(f'Student "{name}" registered successfully.', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('No face detected. Try again.', 'error')
        else:
            flash('Failed to capture frame. Check your webcam and try again.', 'error')
    return render_template('register.html')

@app.route('/student/<int:student_id>')
@login_required
def student_profile(student_id):
    student = db.get_student(student_id)
    if not student:
        return redirect(url_for('dashboard'))
    attendance_history = db.get_student_attendance(student_id)
    return render_template('student_profile.html', student=student, attendance=attendance_history)

@app.route('/student/<int:student_id>/data')
@login_required
def student_profile_data(student_id):
    student = db.get_student(student_id)
    if not student:
        return jsonify({'error': 'Student not found'}), 404
    attendance_history = db.get_student_attendance(student_id)
    return jsonify({
        'student': student,
        'attendance': attendance_history,
        'attendance_count': len(attendance_history)
    })

@app.route('/student/<int:student_id>/delete', methods=['POST'])
@login_required
def delete_student(student_id):
    student = db.get_student(student_id)
    if not student:
        flash('Student profile not found.', 'error')
        return redirect(url_for('dashboard'))

    if db.delete_student(student_id):
        load_students()
        session.pop('last_capture', None)
        flash(f'Student "{student["name"]}" was deleted.', 'success')
    else:
        flash('Unable to delete student profile.', 'error')
    return redirect(url_for('dashboard'))


def gen():
    """Generator for video feed with face recognition and attendance marking."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    marked_today = db.get_marked_student_ids_today()
    pending_marks = set()
    frame_count = 0
    recognized_faces = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            recognized_faces = []
            small_frame = cv2.resize(frame, (0, 0), fx=RECOGNITION_SCALE, fy=RECOGNITION_SCALE)
            rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = detect_faces_fast(small_frame, frame_count)
            face_encodings = face_recognition.face_encodings(rgb, face_locations)

            for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                name = "Unknown"
                if len(known_face_embeddings):
                    distances = face_recognition.face_distance(known_face_embeddings, encoding)
                    best_match_index = int(np.argmin(distances))
                    if distances[best_match_index] < MATCH_THRESHOLD:
                        student_id = known_face_ids[best_match_index]
                        name = known_face_names[best_match_index]
                        if student_id not in marked_today and student_id not in pending_marks:
                            pending_marks.add(student_id)
                            marked_today.add(student_id)
                            mark_attendance_async(student_id)

                scale = int(1 / RECOGNITION_SCALE)
                recognized_faces.append((top * scale, right * scale, bottom * scale, left * scale, name))

        for top, right, bottom, left, name in recognized_faces:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ret:
            continue
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
    cap.release()

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/export_csv')
@login_required
def export_csv():
    csv_data = db.export_csv()
    output = io.BytesIO()
    output.write(csv_data.encode('utf-8'))
    output.seek(0)
    return send_file(output, mimetype='text/csv', as_attachment=True, download_name='attendance.csv')

@app.route('/export_xlsx')
@login_required
def export_xlsx():
    output = db.export_xlsx()
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='attendance.xlsx'
    )

if __name__ == '__main__':
    load_students()  # Load on startup
    app.run(debug=True, host='0.0.0.0', port=5000)
