from datetime import datetime, timedelta

import cv2
import face_recognition

from app import db


class Recognizer:
    @staticmethod
    def recognize(frame):
        global LAST_SAVE
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_labels = []
        names = []
        for face_location, face_encoding in zip(face_locations, face_encodings):
            user_data, distance = db.lookup_known_face(face_encoding)
            top, right, bottom, left = face_location
            photo = small_frame[top:bottom, left:right]
            if user_data is not None:
                face_label = "{0}".format(user_data['name'])
                names.append(user_data['name'])
                db.update_photo(user_data['id'], photo)
            else:
                id = db.register_new_face(face_encoding, photo)
                face_label = "New person: {0}".format(id)
                names.append(id)

            face_labels.append(face_label)

        if datetime.now() - LAST_SAVE > timedelta(seconds=10):
            for label in names:
                db.save_logs(dict(label=label, time=datetime.now()))
            LAST_SAVE = datetime.now()

        return face_locations, face_encodings, face_labels
