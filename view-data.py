import sqlite3
import numpy as np

def list_database_contents():
    conn = sqlite3.connect('faces.db')
    cursor = conn.cursor()

    # List all persons
    cursor.execute("SELECT * FROM persons")
    persons = cursor.fetchall()
    print("Persons:")
    for person in persons:
        print(f"ID: {person[0]}, Name: {person[1]}")

    # List all face encodings
    cursor.execute("SELECT * FROM face_encodings")
    face_encodings = cursor.fetchall()
    print("\nFace Encodings:")
    for encoding in face_encodings:
        person_id, face_encoding = encoding[1], np.frombuffer(encoding[2], dtype=np.float64)
        print(f"ID: {encoding[0]}, Person ID: {person_id}, Face Encoding: {face_encoding}")

    conn.close()

if __name__ == "__main__":
    list_database_contents()
