from flask import Flask, request, render_template, redirect, url_for, Response
import os
from werkzeug.utils import secure_filename

# Reduce TensorFlow verbosity: set before importing modules that load TF.
# 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR. We set 2 to hide INFO messages.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from sound import get_percentages

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
ALLOWED_EXTENSIONS = {"wav", "mp3", "mp4", "flac", "ogg", "m4a", "mpeg"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/favicon.ico')
def favicon():
    # Return a minimal SVG favicon to avoid browser 404 and extra file.
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><rect width="16" height="16" fill="#4CAF50"/></svg>'
    return Response(svg, mimetype='image/svg+xml')


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file part")
        f = request.files["file"]
        if f.filename == "":
            return render_template("index.html", error="No selected file")
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            dest = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            f.save(dest)

            try:
                results = get_percentages(dest, top_n=15)
                return render_template("result.html", results=results)
            except Exception as e:
                return render_template("index.html", error=f"Analysis failed: {str(e)}")
        else:
            return render_template("index.html", error="Unsupported file type")

    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze_api():
    """API endpoint that accepts a single file upload and returns JSON results."""
    if "file" not in request.files:
        return {"error": "No file part"}, 400
    f = request.files["file"]
    if f.filename == "":
        return {"error": "No selected file"}, 400
    if not allowed_file(f.filename):
        return {"error": "Unsupported file type"}, 400

    filename = secure_filename(f.filename)
    dest = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    f.save(dest)

    try:
        results = get_percentages(dest, top_n=15)
        return {"results": results}
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}, 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
