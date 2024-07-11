import cv2
import face_recognition
import sqlite3
import os
def create_database():
    conn = sqlite3.connect('faces.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_encodings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            face_encoding BLOB NOT NULL,
            FOREIGN KEY (person_id) REFERENCES persons (id)
        )
    ''')
    conn.commit()
    conn.close()

def insert_face_encoding(person_id, face_encoding):
    conn = sqlite3.connect('faces.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO face_encodings (person_id, face_encoding) VALUES (?, ?)", (person_id, face_encoding.tobytes()))
    conn.commit()
    conn.close()

def get_or_create_person(name):
    conn = sqlite3.connect('faces.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM persons WHERE name = ?", (name,))
    result = cursor.fetchone()
    if result:
        person_id = result[0]
    else:
        cursor.execute("INSERT INTO persons (name) VALUES (?)", (name,))
        person_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return person_id

def capture_faces():
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit and 's' to save the face encoding.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Press 'q' to quit the application
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Press 's' to save the face encoding
        if cv2.waitKey(1) & 0xFF == ord('s'):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            if face_encodings:
                face_encoding = face_encodings[0]
                name = input("Enter the name of the person: ")
                person_id = get_or_create_person(name)
                insert_face_encoding(person_id, face_encoding)
                print(f"Face encoding for {name} saved.")
            else:
                print("No face detected. Please try again.")

    cap.release()
    cv2.destroyAllWindows()

def process_images_from_folders(base_folder_path):
    for person_name in os.listdir(base_folder_path):
        person_folder = os.path.join(base_folder_path, person_name)
        if os.path.isdir(person_folder):
            person_id = get_or_create_person(person_name)
            for filename in os.listdir(person_folder):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image_path = os.path.join(person_folder, filename)
                    image = cv2.imread(image_path)
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_image)
                    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

                    if face_encodings:
                        face_encoding = face_encodings[0]
                        insert_face_encoding(person_id, face_encoding)
                        print(f"Face encoding for {person_name} from {filename} saved.")
                    else:
                        print(f"No face detected in {filename}.")

if __name__ == "__main__":
    create_database()
    capture_faces()
    # process_images_from_folders('')
