import cv2 
import mediapipe as mp

mpFace = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils

webcam = cv2.VideoCapture(0)

with mpFace.FaceDetection(model_selection=0, min_detection_confidence=0.5) as faces:
  while webcam.isOpened():
    frame_status, frame = webcam.read()

    if not frame_status:
      print("Camera failed.")
      break

    # To improve performance, temporarily mark the image as not writeable to pass by reference.
    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    detectedFaces = faces.process(frame)

    # Draw the face detection annotations on the image.
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if detectedFaces.detections:
      for face in detectedFaces.detections:
        mpDraw.draw_detection(frame, face)

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Detection', cv2.flip(frame, 1))

    if cv2.waitKey(5) & 0xFF == 27:
      break

webcam.release()
cv2.destroyAllWindows()