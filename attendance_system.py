import os
import sqlite3
from datetime import date
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import numpy as np


def get_db_connection(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = "attendance.db") -> None:
    db_path = db_path or "attendance.db"
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = get_db_connection(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            student_id TEXT UNIQUE NOT NULL,
            face_image_path TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            attendance_date TEXT NOT NULL,
            marked_at TEXT NOT NULL,
            face_image_path TEXT,
            FOREIGN KEY(student_id) REFERENCES students(student_id),
            UNIQUE(student_id, attendance_date)
        )
        """
    )
    conn.commit()
    conn.close()


def add_student(db_path: str, name: str, student_id: str, face_image_path: Optional[str] = None) -> str:
    init_db(db_path)
    conn = get_db_connection(db_path)
    existing = conn.execute("SELECT student_id FROM students WHERE student_id = ?", (student_id,)).fetchone()
    if existing:
        conn.execute(
            "UPDATE students SET name = ?, face_image_path = COALESCE(?, face_image_path) WHERE student_id = ?",
            (name, face_image_path, student_id),
        )
        conn.commit()
        conn.close()
        return student_id

    conn.execute(
        "INSERT INTO students (name, student_id, face_image_path) VALUES (?, ?, ?)",
        (name, student_id, face_image_path),
    )
    conn.commit()
    conn.close()
    return student_id


def list_students(db_path: str) -> List[Dict[str, object]]:
    init_db(db_path)
    conn = get_db_connection(db_path)
    rows = conn.execute("SELECT student_id, name, face_image_path FROM students ORDER BY name").fetchall()
    conn.close()
    return [dict(row) for row in rows]


def detect_and_crop_face(image: np.ndarray) -> Optional[np.ndarray]:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    face = image[y : y + h, x : x + w]
    return cv2.resize(face, (200, 200))


def save_face_image(db_path: str, student_id: str, image: np.ndarray) -> Optional[str]:
    face = detect_and_crop_face(image)
    if face is None:
        return None

    faces_dir = Path(os.path.dirname(db_path) or ".") / "faces"
    faces_dir.mkdir(parents=True, exist_ok=True)
    target_path = faces_dir / f"{student_id}.jpg"
    cv2.imwrite(str(target_path), face)
    return str(target_path)


def mark_attendance(db_path: str, student_id: str, face_image_path: Optional[str] = None) -> bool:
    init_db(db_path)
    conn = get_db_connection(db_path)
    student = conn.execute("SELECT student_id FROM students WHERE student_id = ?", (student_id,)).fetchone()
    if not student:
        conn.close()
        raise ValueError(f"Student {student_id} not found")

    today = date.today().isoformat()
    existing = conn.execute(
        "SELECT 1 FROM attendance WHERE student_id = ? AND attendance_date = ?",
        (student_id, today),
    ).fetchone()
    if existing:
        conn.close()
        return False

    conn.execute(
        "INSERT INTO attendance (student_id, attendance_date, marked_at, face_image_path) VALUES (?, ?, ?, ?)",
        (student_id, today, str(date.today()), face_image_path),
    )
    conn.commit()
    conn.close()
    return True


def get_attendance_report(db_path: str, attendance_date: Optional[str] = None) -> List[Dict[str, object]]:
    init_db(db_path)
    conn = get_db_connection(db_path)
    target_date = attendance_date or date.today().isoformat()
    rows = conn.execute(
        """
        SELECT s.student_id, s.name, a.attendance_date, a.marked_at
        FROM students s
        LEFT JOIN attendance a
            ON a.student_id = s.student_id AND a.attendance_date = ?
        ORDER BY s.name
        """,
        (target_date,),
    ).fetchall()
    conn.close()
    report = []
    for row in rows:
        if row["attendance_date"] is None:
            continue
        report.append(
            {
                "student_id": row["student_id"],
                "name": row["name"],
                "attendance_date": row["attendance_date"],
                "marked_at": row["marked_at"],
            }
        )
    return report
