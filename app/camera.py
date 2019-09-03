import cv2

from app.recognition import Recognizer


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        status, image = self.video.read()
        bboxes, encodings, labels = Recognizer.recognize(image)
        frame = self.draw_faces(image, bboxes, labels)
        return self.get_frame_bytes(frame)

    @staticmethod
    def draw_faces(frame, bboxes, labels):
        for (top, right, bottom, left), face_label in zip(bboxes, labels):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, face_label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        return frame

    @staticmethod
    def get_frame_bytes(frame):
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
