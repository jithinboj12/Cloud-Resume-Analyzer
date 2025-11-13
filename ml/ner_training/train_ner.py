import spacy
from spacy.util import minibatch, compounding
import random
import plac
import pathlib

def train(output_dir: str, n_iter: int = 20, train_data_path: str = None):
    if train_data_path is None:
        raise ValueError("Please provide path to training data in spaCy DocBin or JSON format.")
    # load base model
    nlp = spacy.load("en_core_web_sm")
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    # Example: If you need to add labels:
    labels = ["SKILL", "JOB_TITLE", "DEGREE", "ORG", "DATE"]
    for label in labels:
        ner.add_label(label)

    # Load training data
    # For simplicity, we expect a list of (text, {"entities": [(start, end, label), ...]})
    import json
    with open(train_data_path, "r", encoding="utf8") as f:
        train_data = json.load(f)

    optimizer = nlp.resume_training()
    for itn in range(n_iter):
        random.shuffle(train_data)
        losses = {}
        batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
        print(f"Iteration {itn}, Losses: {losses}")

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(output_dir)
    print(f"Saved trained NER model to {output_dir}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--train_data", required=True, help="Path to JSON training data: list of [text, {'entities': [...]}]")
    ap.add_argument("--n_iter", type=int, default=20)
    args = ap.parse_args()
    train(args.output_dir, args.n_iter, args.train_data)
