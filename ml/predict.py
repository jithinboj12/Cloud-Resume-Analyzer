import json
import os
from parser import parse_text
from feature_extractor import extract_features
from scorer import predict as scorer_predict, load_model
from typing import Dict, Any

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

def run_inference_on_text(text: str) -> Dict[str, Any]:
    parsed = parse_text(text)
    feats = extract_features(parsed)
    # attempt to call scorer; handle missing model gracefully
    try:
        score = scorer_predict(feats)
    except FileNotFoundError:
        score = {"label": None, "confidence": None}
    output = {
        "parsed": parsed,
        "features": feats,
        "score": score
    }
    return output


def run_from_file(path: str):
    with open(path, "r", encoding="utf8") as f:
        text = f.read()
    return run_inference_on_text(text)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--text_file", help="Path to plain text resume. If not provided, a sample is used.")
    args = ap.parse_args()
    if args.text_file:
        out = run_from_file(args.text_file)
    else:
        sample = """Alice B
alice@example.com | +91 9876543210

Professional Experience
ML Engineer - Acme Corp Feb 2019 - Sep 2022
- Built models using python and pytorch
- Deployed with docker and AWS

Education
B.Tech in Computer Science, XYZ University, 2018

Skills
Python, PyTorch, TensorFlow, Docker, AWS
"""
        out = run_inference_on_text(sample)
    print(json.dumps(out, indent=2))