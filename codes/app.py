from flask import Flask, render_template, request, jsonify
import text_miner

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    request_data = request.form
    text1 = request_data["reference_text"]
    text2 = request_data["candidate_text"]



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
