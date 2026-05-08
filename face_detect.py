import cv2
import mediapipe as mp

def detect_faces(frames):
    faces = []
    
    with mp.solutions.face_detection.FaceDetection(
        model_selection=0, 
        min_detection_confidence=0.5
    ) as mp_face:
        
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = mp_face.process(rgb)

            if result.detections:
                for detection in result.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape

                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    bw = int(bbox.width * w)
                    bh = int(bbox.height * h)

                    x = max(0, x)
                    y = max(0, y)
                    bw = min(bw, w - x)
                    bh = min(bh, h - y)

                    face = frame[y:y+bh, x:x+bw]

                    if face.size != 0:
                        face = cv2.resize(face, (224, 224))
                        faces.append(face)

    return faces