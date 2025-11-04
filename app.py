from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import onnxruntime as ort

app = Flask(__name__)

# Load models
print("🔄 Loading models...")
face_net = cv2.dnn.readNetFromCaffe("RFB-320.prototxt", "RFB-320.caffemodel")
emotion_model = ort.InferenceSession("emotion-ferplus-8.onnx")
emotions = ["Neutral", "Happiness", "Surprise", "Sadness", "Anger", "Disgust", "Fear", "Contempt"]
print("✅ Models loaded successfully!")

print("Model input shape:", emotion_model.get_inputs()[0].shape)

# ---- FACE DETECTION ----
def detect_faces_in_frame(image):
    """Detect faces using OpenCV DNN model"""
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            faces.append(image[y1:y2, x1:x2])
    return faces


# ---- PREPROCESS ----
def preprocess_face(face):
    try:
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))
        blob = resized.reshape(1, 1, 64, 64).astype("float32")  # match ONNX input
        return blob
    except Exception as e:
        print("❌ Preprocessing error:", e)
        return None


# ---- EMOTION PREDICTION ----
def predict_emotion(face):
    blob = preprocess_face(face)
    if blob is None:
        return "Unknown"

    try:
        input_name = emotion_model.get_inputs()[0].name
        output_name = emotion_model.get_outputs()[0].name
        preds = emotion_model.run([output_name], {input_name: blob})[0]
        emotion = emotions[int(np.argmax(preds))]
        return emotion
    except Exception as e:
        print("❌ Prediction error:", e)
        return "Unknown"


# ---- ROUTES ----
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        img_data = base64.b64decode(data["image"].split(",")[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        faces = detect_faces_in_frame(frame)
        results = []

        for face in faces:
            emotion = predict_emotion(face)
            results.append(emotion)

        if not results:
            return jsonify({"error": "No face detected"}), 200

        return jsonify({"emotions": results})

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🚀 Starting Flask app...")
    app.run(host="0.0.0.0", port=5000)
