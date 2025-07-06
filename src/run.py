import os
import random
import argparse
from tqdm import tqdm
from datasets import load_dataset
import fasttext

def stream_sample_fineweb(start=50000, num_samples=5000, min_words=5, seed=42):
    ds = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
    rng = random.Random(seed)
    samples = []

    print(f"Streaming fineweb to collect {num_samples} samples...")

    i = 0
    for item in ds:
        i+=1
        if i == start:
            break
    
    for item in tqdm(ds, desc="Sampling"):
        try:            
            text = item["text"].strip().replace("\n", " ")
            if len(text.split()) >= min_words:
                samples.append(text)
            if len(samples) >= num_samples:
                break
        except:
            continue

    rng.shuffle(samples)
    return samples

def infer_labels(model_path, texts):
    model = fasttext.load_model(model_path)
    labeled_texts = []

    for text in tqdm(texts, desc="Predicting"):
        label = model.predict(text)[0][0]
        labeled_texts.append(f"{label} {text}")

    return labeled_texts

def save_to_file(lines, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    print(f"Saved {len(lines)} lines to {filepath}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model.bin', help="Path to fasttext model")
    parser.add_argument('--output', type=str, default='result/predicted_samples.txt', help="Path to save labeled output")
    parser.add_argument('--samples', type=int, default=5000, help="Number of fineweb samples to download")
    args = parser.parse_args()

    texts = stream_sample_fineweb(num_samples=args.samples)
    labeled = infer_labels(args.model, texts)
    save_to_file(labeled, args.output)

if __name__ == "__main__":
    main()
