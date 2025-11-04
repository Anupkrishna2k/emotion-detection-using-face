from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import onnxruntime as ort
#from face_timestamp import detect_faces_in_frame  # use the developer's face detection

app = Flask(__name__)

# Load models once at startup
print("Loading face detection and emotion recognition models...")

face_net = cv2.dnn.readNetFromCaffe("RFB-320.prototxt", "RFB-320.caffemodel")
emotion_model = ort.InferenceSession("emotion-ferplus-8.onnx")
emotions = [
    "Neutral", "Happiness", "Surprise", "Sadness",
    "Anger", "Disgust", "Fear", "Contempt"
]

print("✅ Models loaded successfully.")

# Function to preprocess face ROI for emotion model
def preprocess_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (64, 64))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)
    face = np.expand_dims(face, axis=0)
    return face

# Home route
@app.route("/")
def index():
    return render_template("index.html")

# Predict route (called from browser)
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get image from request
        data = request.get_json()
        img_data = data["image"]
        img_data = img_data.split(",")[1]
        nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Detect faces
        faces = detect_faces_in_frame(frame, face_net)
        results = []

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            if face_roi.size == 0:
                continue

            processed_face = preprocess_face(face_roi)
            ort_inputs = {emotion_model.get_inputs()[0].name: processed_face}
            ort_outs = emotion_model.run(None, ort_inputs)
            emotion_idx = np.argmax(ort_outs[0])
            emotion = emotions[emotion_idx]

            results.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "emotion": emotion})

        return jsonify({"results": results})

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)})

# Health check route for Render
@app.route("/health")
def health():
    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
