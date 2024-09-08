import cv2
import face_recognition
import os

def load_known_face_encodings(user_folder):
    known_face_encodings = []
    known_face_names = []

    for user_folder_name in os.listdir(user_folder):
        user_folder_path = os.path.join(user_folder, user_folder_name)
        if os.path.isdir(user_folder_path):
            for image_file in os.listdir(user_folder_path):
                image_path = os.path.join(user_folder_path, image_file)
                image = face_recognition.load_image_file(image_path)

                face_encodings = face_recognition.face_encodings(image)

                if face_encodings:
                    known_face_encodings.append(face_encodings[0])
                    known_face_names.append(user_folder_name)
                else:
                    print(f"No face found in image: {image_file}")

    print(f"Loaded {len(known_face_encodings)} known face encodings.")
    return known_face_encodings, known_face_names

image_db_folder = "image_db/Person"

known_face_encodings, known_face_names = load_known_face_encodings(image_db_folder)

cam = cv2.VideoCapture(0)
face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    print(f"Detected {len(face_encodings)} face(s) in the frame.")

    face_names = []

    for face_encoding in face_encodings:
        if known_face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            print(f"Face distances: {face_distances}")

            if face_distances.size > 0:
                best_match_index = face_distances.argmin()
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                else:
                    name = "Unknown"
            else:
                name = "Unknown"
        else:
            name = "Unknown"

        face_names.append('Face Recognized: ' + name)

    # Draw rectangles and labels on recognized faces
    for (x, y, w, h), name in zip(faces, face_names):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
