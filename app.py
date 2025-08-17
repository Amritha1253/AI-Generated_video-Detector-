import os
import random
from flask import Flask, render_template, request, redirect, url_for

# Create Flask app
app = Flask(__name__)

# Where uploaded files will be saved
UPLOAD_FOLDER = os.path.join("static", "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Dummy Model Function ---
# Replace this with your actual deepfake detection model
def detect_video(filepath):
    """
    Fake detector: returns random result + confidence.
    Replace with your trained model prediction.
    """
    confidence = random.uniform(0.4, 0.99)  # fake probability between 40% and 99%
    if confidence > 0.6:
        return "AI-Generated (Deepfake)", confidence
    else:
        return "Real Video", confidence


# --- Routes ---
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get uploaded file
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            # Save uploaded video
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            # Run detection
            result, confidence = detect_video(filepath)

            # Render with results
            return render_template(
                "index.html",
                result=result,
                confidence=confidence,
                filename=file.filename,
            )

    # Default GET request -> load empty page
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
