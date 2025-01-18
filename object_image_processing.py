import cv2

# def load_object():

cam = cv2.VideoCapture(0)
people_cascade = cv2.CascadeClassifier('lib/xml_lib/haarcascade_fullbody.xml')

recognized_people = []

while True:

    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    people_objects = people_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in people_objects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        recognized_people.append('People Recognized')

    cv2.imshow("People Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()