import cv2
import dlib

# Load a pre-trained face detector model from Dlib
face_detector = dlib.get_frontal_face_detector()

# Load a pre-trained face recognition model from Dlib
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Initialize a dictionary to store known faces
known_faces = {
    "Person1": dlib.face_encodings(dlib.load_rgb_image("person1.jpg"))[0],
    "Person2": dlib.face_encodings(dlib.load_rgb_image("person2.jpg"))[0],
}

# Open a video capture object (0 is the default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Detect faces in the frame
    faces = face_detector(frame)

    for face in faces:
        # Get the face landmarks
        landmarks = shape_predictor(frame, face)
        
        # Recognize the face
        face_encoding = face_recognizer.compute_face_descriptor(frame, landmarks)

        # Compare the face encoding with known faces
        for name, known_face_encoding in known_faces.items():
            match = dlib.face_distance([known_face_encoding], face_encoding)

            if match[0] < 0.6:  # Adjust the threshold as needed
                cv2.putText(frame, name, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press Esc to exit
        break

cap.release()
cv2.destroyAllWindows()
