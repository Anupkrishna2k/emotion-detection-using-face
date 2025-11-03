const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const resultDiv = document.getElementById("result");

navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => video.srcObject = stream)
  .catch(err => alert("Camera access denied: " + err));

async function captureAndSend() {
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const dataUrl = canvas.toDataURL("image/jpeg");

  const res = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: dataUrl })
  });

  const data = await res.json();
  if (data.results && data.results.length > 0) {
    const emotions = data.results.map(r => r.emotion).join(", ");
    resultDiv.textContent = "Detected: " + emotions;
  } else {
    resultDiv.textContent = "No face detected.";
  }
}

setInterval(captureAndSend, 1500);
