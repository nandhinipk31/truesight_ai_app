import json

def log_detection(filename, result, confidence):
    try:
        with open("detection_history.json", "r") as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []

    history.append({
        "filename": filename,
        "result": result,
        "confidence": float(confidence)
    })

    with open("detection_history.json", "w") as f:
        json.dump(history, f, indent=4)