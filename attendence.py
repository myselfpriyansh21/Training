import os
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from attendance_system import add_student, get_attendance_report, init_db, list_students, mark_attendance, save_face_image

DB_PATH = os.path.join(os.path.dirname(__file__), "attendance.db")
init_db(DB_PATH)

st.set_page_config(page_title="Attendance App", layout="centered")
st.title("Attendance Management System")
st.caption("Register students, capture faces with OpenCV, and mark attendance in SQLite")

page = st.sidebar.radio("Navigation", ["Register Student", "Mark Attendance", "Attendance Report"])

if page == "Register Student":
    st.subheader("Register a student")
    with st.form("register_student"):
        name = st.text_input("Student name")
        student_id = st.text_input("Student ID")
        image_bytes = st.camera_input("Capture student face")
        submitted = st.form_submit_button("Save student")

    if submitted:
        if not name or not student_id:
            st.error("Please enter both name and student ID")
        elif image_bytes is None:
            st.error("Please capture a face image first")
        else:
            image_array = np.frombuffer(image_bytes.getvalue(), np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            face_path = save_face_image(DB_PATH, student_id, image)
            if face_path is None:
                st.error("No face detected. Please try again with a clearer view")
            else:
                add_student(DB_PATH, name, student_id, face_image_path=face_path)
                st.success(f"Student {name} registered successfully")

elif page == "Mark Attendance":
    st.subheader("Mark attendance")
    students = list_students(DB_PATH)
    if not students:
        st.warning("No students registered yet. Register one from the sidebar first.")
        st.stop()

    student_options = {student["student_id"]: student["name"] for student in students}
    selected_student_id = st.selectbox("Choose student", list(student_options.keys()), format_func=lambda sid: f"{student_options[sid]} ({sid})")
    image_bytes = st.camera_input("Capture attendance face")

    if st.button("Mark attendance"):
        if image_bytes is None:
            st.error("Please capture a face image first")
        else:
            image_array = np.frombuffer(image_bytes.getvalue(), np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            face_path = save_face_image(DB_PATH, selected_student_id, image)
            if face_path is None:
                st.error("No face detected. Please try again")
            else:
                marked = mark_attendance(DB_PATH, selected_student_id, face_image_path=face_path)
                if marked:
                    st.success(f"Attendance marked for {student_options[selected_student_id]}")
                else:
                    st.info("Attendance already marked for this student today")

else:
    st.subheader("Today's attendance report")
    report = get_attendance_report(DB_PATH)
    if report:
        st.dataframe(report, use_container_width=True)
    else:
        st.info("No attendance recorded yet")
