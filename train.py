from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

from common import LABELS, encode, read_csv, tune_thresholds


class Chunks(Dataset):
    def __init__(self, encodings, mapping, labels=None):
        self.encodings, self.mapping, self.labels = encodings, np.asarray(mapping), labels

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, i):
        item = {key: torch.tensor(value[i]) for key, value in self.encodings.items()}
        item["document_index"] = torch.tensor(self.mapping[i])
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[self.mapping[i]]).float()
        return item


def tokenize(tokenizer, texts, max_length, stride):
    encoded = tokenizer(texts, truncation=True, max_length=max_length, stride=stride,
                        return_overflowing_tokens=True, padding="max_length")
    return encoded, encoded.pop("overflow_to_sample_mapping")


@torch.no_grad()
def predict_probability(model, loader, document_count, device, pooling):
    model.eval()
    sums = np.zeros((document_count, len(LABELS)), dtype=np.float64)
    maxima = np.zeros_like(sums)
    counts = np.zeros(document_count, dtype=np.int64)
    for batch in loader:
        indices = batch.pop("document_index").numpy()
        batch.pop("labels", None)
        inputs = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            probability = torch.sigmoid(model(**inputs).logits).float().cpu().numpy()
        for row, index in enumerate(indices):
            sums[index] += probability[row]
            maxima[index] = np.maximum(maxima[index], probability[row])
            counts[index] += 1
    return maxima.astype(np.float32) if pooling == "max" else (sums / counts[:, None]).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="klue/roberta-large")
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--pooling", choices=["max", "mean"], default="max")
    parser.add_argument("--pos_weight_power", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("GPU 런타임이 아닙니다. Colab 런타임을 GPU로 변경하세요.")
    device = torch.device("cuda")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    frame = read_csv(args.train_csv)
    labels = encode(frame["symptom_str"])
    split = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    train_index, dev_index = next(split.split(frame, labels))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(LABELS), problem_type="multi_label_classification"
    ).to(device)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    train_encoded, train_map = tokenize(tokenizer, frame.iloc[train_index]["text"].tolist(), args.max_length, args.stride)
    dev_encoded, dev_map = tokenize(tokenizer, frame.iloc[dev_index]["text"].tolist(), args.max_length, args.stride)
    train_labels, dev_labels = labels[train_index], labels[dev_index]
    train_loader = DataLoader(Chunks(train_encoded, train_map, train_labels), batch_size=args.batch_size,
                              shuffle=True, pin_memory=True, num_workers=2)
    dev_loader = DataLoader(Chunks(dev_encoded, dev_map, dev_labels), batch_size=args.batch_size * 2,
                            shuffle=False, pin_memory=True, num_workers=2)

    positives = train_labels.sum(0)
    ratio = (len(train_labels) - positives) / np.maximum(positives, 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(ratio ** args.pos_weight_power, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    updates_per_epoch = int(np.ceil(len(train_loader) / args.gradient_accumulation))
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_updates * args.warmup_ratio), total_updates)
    scaler = torch.amp.GradScaler("cuda")
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    best_score, best_thresholds, best_epoch = -1.0, None, 0

    for epoch in range(1, args.epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); running = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch.pop("document_index")
            targets = batch.pop("labels").to(device, non_blocking=True)
            inputs = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = criterion(model(**inputs).logits, targets) / args.gradient_accumulation
            scaler.scale(loss).backward(); running += float(loss) * args.gradient_accumulation
            if step % args.gradient_accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            if step % 200 == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={running / step:.4f}")

        probability = predict_probability(model, dev_loader, len(dev_index), device, args.pooling)
        thresholds, score = tune_thresholds(probability, dev_labels)
        print(f">>> epoch={epoch} internal Macro F1={score:.4f}")
        if score > best_score:
            best_score, best_thresholds, best_epoch = score, thresholds, epoch
            model.save_pretrained(output, safe_serialization=True)
            tokenizer.save_pretrained(output)
            (output / "metadata.json").write_text(json.dumps({
                "model_name": args.model_name, "labels": LABELS, "max_length": args.max_length,
                "stride": args.stride, "pooling": args.pooling, "thresholds": thresholds.tolist(),
                "best_epoch": epoch, "internal_dev_macro_f1": score, "seed": args.seed
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        torch.cuda.empty_cache()
    print(f"완료: best epoch={best_epoch}, internal Macro F1={best_score:.4f}, saved={output}")


if __name__ == "__main__":
    main()
