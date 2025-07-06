import os
import random
from datasets import load_dataset
from tqdm import tqdm

def stream_sample_texts(dataset_name, split, key, num_samples, seed=42):
    ds = load_dataset(dataset_name, split=split, streaming=True)
    valid = []

    rng = random.Random(seed)
    print(f"Streaming from {dataset_name}...")
    for item in tqdm(ds, desc=f"Sampling {dataset_name}"):
        try:
            text = item[key].strip().replace('\n', ' ')
            if len(text.split()) >= 5:
                valid.append(text)
            if len(valid) >= num_samples:
                break
        except:
            continue

    rng.shuffle(valid)
    return valid

def save_samples(texts, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for t in texts:
            f.write(t + "\n")

def main():
    pos = stream_sample_texts("open-web-math/open-web-math", "train", "text", 30000)
    neg = stream_sample_texts("HuggingFaceFW/fineweb", "train", "text", 30000)

    save_samples(pos, "data/openwebmath_samples.txt")
    save_samples(neg, "data/fineweb_samples.txt")

    print(f"Saved {len(pos)} math texts to data/openwebmath_samples.txt")
    print(f"Saved {len(neg)} other texts to data/fineweb_samples.txt")

if __name__ == "__main__":
    main()