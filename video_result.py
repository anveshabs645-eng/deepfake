import cv2
from video_frames import extract_frames
from face_detect import detect_faces
from video_model import predict_faces

# -----------------------------
# ENHANCED DEEPFAKE SCORE
# -----------------------------
def enhanced_score(face, cnn_score):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    # Sharpness (deepfakes often smoother)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Normalize sharpness
    sharpness_score = min(sharpness / 1000, 1.0)

    # Combine CNN + artifact detection
    final_score = 0.6 * cnn_score + 0.4 * (1 - sharpness_score)

    return final_score


# -----------------------------
# MAIN VIDEO ANALYSIS FUNCTION
# -----------------------------
def analyze_video(video_path):
    print("Extracting frames...")
    frames = extract_frames(video_path)

    print("Detecting faces...")
    faces = detect_faces(frames)

    if len(faces) == 0:
        return {"error": "No face detected"}

    print(f"Faces detected: {len(faces)}")

    print("Running deepfake detection...")

    scores = []

    # 🔥 FINAL CORRECT LOOP
    for face in faces:
        cnn_scores = predict_faces([face])   # EfficientNet prediction
        score = enhanced_score(face, cnn_scores[0])  # combine with artifacts
        scores.append(score)

    # -----------------------------
    # FINAL DECISION
    # -----------------------------
    avg_score = float(sum(scores) / len(scores))

    if avg_score < 0.4:
        result = "Real"
    elif avg_score < 0.6:
        result = "Suspicious"
    else:
        result = "Fake"

    return {
        "score": round(avg_score, 2),
        "result": result,
        "faces_detected": len(faces)
    }


# -----------------------------
# TEST RUN
# -----------------------------
if __name__ == "__main__":
    video_path = "sample.mp4"   # your video file
    output = analyze_video(video_path)

    print("\nFINAL RESULT:")
    print(output)