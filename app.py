# app.py
from flask import Flask, request, jsonify
from model.predict import predict_from_csv
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files are allowed"}), 400

    try:
        logging.info("Received file: %s", file.filename)
        result = predict_from_csv(file)
        return jsonify(result)
    except Exception as e:
        logging.exception("Prediction failed:")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
