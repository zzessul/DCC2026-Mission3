from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import decode, read_csv
from train import Chunks, predict_probability, tokenize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint_dir)
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    frame = read_csv(args.input_csv, require_labels=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
    encoded, mapping = tokenize(tokenizer, frame["text"].tolist(), metadata["max_length"], metadata["stride"])
    loader = DataLoader(Chunks(encoded, mapping), batch_size=args.batch_size, shuffle=False,
                        pin_memory=device.type == "cuda", num_workers=2)
    probability = predict_probability(model, loader, len(frame), device, metadata["pooling"])
    prediction = probability >= np.asarray(metadata["thresholds"])
    result = pd.DataFrame({"label_file": frame["label_file"], "symptom_str": decode(prediction)})
    output = Path(args.output_csv); output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {output} ({len(result)}건)")


if __name__ == "__main__":
    main()
