import cv2
import numpy as np
import torch
import time
import sqlite3
import face_recognition
from facenet_pytorch import MTCNN
from sort.sort import Sort

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Person:
    def __init__(self, person_id, face):
        self.person_id = person_id
        self.face = face
        self.name = "Unknown"
        self.prob = 0.0
        self.timeout = 10
        self.show_face = None
        self.pre_name = "Unknown"
        self.show_prob = 0.0

    def update(self, face):
        self.face = face
        self.timeout = 10

class Detector:
    def __init__(self, min_width=0, list_len=5):
        self.stream = None
        self.list_len = list_len
        self.list_img_size = 0
        self.tl = (0, 0)
        self.br = (0, 0)
        self.scale = 1
        self.persons = {}
        self.known_person_ids,self.known_face_encodings, self.known_face_names= self.load_known_faces()

    def load_known_faces(self):
        conn = sqlite3.connect('faces.db')
        cursor = conn.cursor()
        cursor.execute("""
            SELECT persons.id, persons.name, face_encodings.face_encoding 
            FROM face_encodings 
            JOIN persons ON face_encodings.person_id = persons.id
        """)
        rows = cursor.fetchall()
        known_face_encodings = []
        known_face_names = []
        known_person_ids = []
        for row in rows:
            person_id = row[0]
            name = row[1]
            encoding = np.frombuffer(row[2], dtype=np.float64)
            known_face_encodings.append(encoding)
            known_face_names.append(name)
            known_person_ids.append(person_id)
        conn.close()
        return known_person_ids, known_face_encodings, known_face_names

    def recognize_face(self, face_encoding):
        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            return self.known_person_ids[best_match_index], self.known_face_names[best_match_index]
        return None, "Unknown"

    def detect(self, stream_path, scale=1, roi=None):
        self.stream = cv2.VideoCapture(stream_path)
        ok, frame = self.stream.read()
        if not ok:
            return ok, None

        h, w = frame.shape[:2]
        self.list_img_size = h // self.list_len
        self.scale = scale

        if roi is not None:
            self.tl = roi[0]
            self.br = roi[1]
        else:
            self.tl, self.br = (0, 0), (w, h)

        tracker = Sort(max_age=0, min_hits=0)

        mtcnn = MTCNN(
            image_size=160,
            min_face_size=50,
            thresholds=[0.6, 0.7, 0.87],
            margin=20,
            post_process=False,
            device=device
        )

        frame_count = 0
        start_time = time.time()

        try:
            while True:
                ok, frame = self.stream.read()
                if not ok:
                    return ok, None

                frame_count += 1

                cropped_frame = frame[self.tl[1]:self.br[1], self.tl[0]:self.br[0]]
                rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)

                detections = np.empty((0, 5))

                boxes, probs = mtcnn.detect(rgb)
                if boxes is not None:
                    valid_idx = [True if prob > 0.99 and all(box > 0) else False
                                 for prob, box in zip(probs, boxes)]
                    boxes = boxes[valid_idx]
                    probs = probs[valid_idx]

                    detections = np.concatenate((boxes, probs.reshape(-1, 1)), axis=1)
                    self.track(tracker, frame, cropped_frame, self.tl, 1, detections=detections)

                i = 0
                for index, id_ in enumerate(list(self.persons.keys())):
                    index -= i
                    if self.persons[id_].show_face is not None and self.persons[id_].pre_name != "Unknown":
                        img_ = np.zeros((self.list_img_size, self.list_img_size, 3), dtype=np.uint8)
                        face = cv2.resize(self.persons[id_].show_face, (self.list_img_size, self.list_img_size))

                        tl_list = (index * self.list_img_size, 0)
                        br_list = (index * self.list_img_size + self.list_img_size, self.list_img_size)

                        frame[tl_list[0]:br_list[0], tl_list[1]:br_list[1]] = face

                        cv2.putText(
                            frame,
                            f"{self.persons[id_].pre_name} {self.persons[id_].show_prob:.2f}",
                            (10, br_list[0] - 20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1,
                            color=(255, 255, 255),
                            thickness=2
                        )
                    else:
                        i += 1

                cv2.rectangle(frame, pt1=tuple(self.tl), pt2=tuple(self.br), color=(0, 255, 0), thickness=2)
                yield ok, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = frame_count / elapsed_time
            print(f"FPS: {fps:.2f}")
            self.stream.release()

    def track(self, tracker, frame, cropped_frame, p1, x, detections=np.empty((0, 5))):
        tracked_objs = tracker.update(detections)
        dh, dw = cropped_frame.shape[:2]

        obj_list = []
        for boxes_with_ids in tracked_objs:
            x1, y1, x2, y2, obj_id = boxes_with_ids.astype(int)

            if x1 <= 0 or x2 >= dw or y1 <= 0 or y2 >= dh:
                continue
            
            face = crop((x1, y1, x2, y2), cropped_frame, padding=2)
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_encoding = face_recognition.face_encodings(face_rgb)
   
            if face_encoding:
                face_encoding = face_encoding[0]
                person_id, name = self.recognize_face(face_encoding)
                if obj_id not in self.persons:
                    self.persons[obj_id] = Person(person_id, face)
                    self.persons[obj_id].person_id = person_id
                    self.persons[obj_id].pre_name = name
                else:
                    self.persons[obj_id].update(face)
                    

            x1, y1 = int(x1 / x) + p1[0], int(y1 / x) + p1[1]
            x2, y2 = int(x2 / x) + p1[0], int(y2 / x) + p1[1]

            tl = (x1, y1 > 25 and y1 - 25 or y1)
            br = (x2, y1 > 25 and y1 or y1 + 25)

            cv2.rectangle(frame, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=2)

            if obj_id not in self.persons.keys():
                self.persons[obj_id] = Person(obj_id, face)
            else:
                self.persons[obj_id].update(face)

            obj_list.append(obj_id)

            cv2.rectangle(frame, pt1=tl, pt2=br, color=(0, 255, 0), thickness=1)
            cv2.putText(
                frame,
                f"{self.persons[obj_id].person_id} {self.persons[obj_id].pre_name}",
                org=(x1 + 5, tl[1] + 20),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.75,
                color=(255, 150, 255),
                thickness=2
            )

        if len(self.persons.keys()) > self.list_len:
            tmp = sorted(self.persons.keys(), key=lambda x: self.persons[x].timeout)
            tmp = tmp[0]
            del self.persons[tmp]

        for id_ in list(self.persons.keys()):
            if id_ not in obj_list:
                if self.persons[id_].timeout < 1:
                    del self.persons[id_]
                else:
                    self.persons[id_].timeout -= 1

def resize(img, scale):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

def crop(box, frame, padding=0):
    return frame[max(0, box[1] - padding):min(box[3] + padding, frame.shape[0]),
                 max(0, box[0] - padding):min(box[2] + padding, frame.shape[1])]

def main():
    stream_path = "test2.mp4"
    detector = Detector()
    detect = detector.detect(stream_path=stream_path, scale=1, roi=None)
    next(detect)

    _, frame = detector.stream.read()
    cv2.namedWindow("Detector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detector", frame.shape[1], frame.shape[0])
    tl_x, tl_y, w, h = cv2.selectROI("Detector", frame, showCrosshair=False)
    detector.tl = (tl_x, tl_y)
    detector.br = (tl_x + w, tl_y + h)

    for ok, frame in detect:
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
