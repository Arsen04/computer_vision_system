import cv2
import os
import time
import src.Enum.FeatureEnum as fe

class Lib:
    IMAGE_DB = "image_db"
    USER_PATH = "\Person"
    EMOTION_PATH = "\Emotion"

    @staticmethod
    def functionality():
        try:
            functionality_choice = int(input(
                "Would you like to  \n "
                "1. register a new user? (Type 1) \n "
                "2. add a new emotion? (Type 2) \n"
            ))

        except ValueError:
            return

        if functionality_choice == fe.FeatureEnum.USER_FEATURE.value:
            new_user = input("Please, enter new user name: ")
            directory_path = os.path.join(os.getcwd(), Lib.IMAGE_DB + Lib.USER_PATH, new_user)
            os.makedirs(directory_path, exist_ok=True)
        elif functionality_choice == fe.FeatureEnum.EMOTION_FEATURE.value:
            new_emotion = input("Please, enter new emotion: ")
            directory_path = os.path.join(os.getcwd(), Lib.IMAGE_DB + Lib.EMOTION_PATH, new_emotion)
            os.makedirs(directory_path, exist_ok=True)
        else:
            print("Invalid input")
            return

        return {
            "functionality_type": functionality_choice,
            "db_path": directory_path
        }

    @staticmethod
    def keyword_processing(frame):
        face_detect = cv2.CascadeClassifier('lib/xml_lib/haarcascade_frontalface_default.xml')
        faces = face_detect.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.putText(frame, "Press 'c' button to capture yourself, or 'q' to exit", (50, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
        cv2.imshow('Frame', frame)

    @staticmethod
    def capturing_image(directory, vid):
        face_detect = cv2.CascadeClassifier('lib/xml_lib/haarcascade_frontalface_default.xml')

        img_counter = 0
        total_images = 30

        while img_counter < total_images:
            ret, frame = vid.read()

            if not ret:
                print("Failed to capture frame!")
                continue

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detect.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:
                print("No face detected!")
                continue

            for (x, y, w, h) in faces:
                face = frame[y:y + h, x:x + w]

                img_name = os.path.join(directory, f'_face_{img_counter}.png')
                gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(img_name, gray_face)
                print(f"Face {img_name} written!")

                img_counter += 1

                if img_counter >= total_images:
                    break

            time.sleep(0.5)

        print(f"Captured {img_counter} face images.")

    @staticmethod
    def open_camera(directory):
        vid = cv2.VideoCapture(0)
        vid.set(3, 400)
        vid.set(4, 400)

        while True:
            ret, frame = vid.read()
            if not ret:
                print("Error: Failed to capture frame")
                break

            Lib.keyword_processing(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            elif cv2.waitKey(1) & 0xFF == ord('c'):
                Lib.capturing_image(directory, vid)
                break
            print("nothing pressed")

        vid.release()
        cv2.destroyAllWindows()

        return vid
