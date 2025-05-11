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
face_detect = cv2.CascadeClassifier('lib/xml_lib/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('lib/xml_lib/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('lib/xml_lib/haarcascade_smile.xml')

frame_count = 0
process_every_n_frames = 15

face_locations = []
face_encodings = []
face_names = []

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        face_image = frame[y:y + h, x:x + w]
        small_face = cv2.resize(face_image, (0, 0), fx=0.25, fy=0.25)
        rgb_small_face = cv2.cvtColor(small_face, cv2.COLOR_BGR2RGB)

    # eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    # smiles = smile_cascade.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=20)

    # for (ex, ey, ew, eh) in eyes:
    #     cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)
    #
    # for (sx, sy, sw, sh) in smiles:
    #     cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 2)
    #     cv2.putText(frame, 'Smiling', (sx, sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    if frame_count % process_every_n_frames == 0:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame, model='hog')
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            name = "Unknown"
            if known_face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                if len(face_distances) > 0:
                    best_match_index = face_distances.argmin()
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
            face_names.append("Face Recognized: " + name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cam.release()
cv2.destroyAllWindows()
