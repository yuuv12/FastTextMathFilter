import os

def format_for_fasttext(pos_path, neg_path, out_train, out_test, test_ratio=0.2, seed=42):
    import random
    pos = [line.strip() for line in open(pos_path, 'r', encoding='utf-8')]
    neg = [line.strip() for line in open(neg_path, 'r', encoding='utf-8')]
    labeled = [("__label__math " + t) for t in pos] + [("__label__other " + t) for t in neg]
    random.Random(seed).shuffle(labeled)

    split = int(len(labeled) * test_ratio)
    test = labeled[:split]
    train = labeled[split:]
    os.makedirs(os.path.dirname(out_train), exist_ok=True)
    with open(out_train, 'w', encoding='utf-8') as f:
        f.write("\n".join(train))
    with open(out_test, 'w', encoding='utf-8') as f:
        f.write("\n".join(test))
    print(f"{len(train)} train and {len(test)} test samples saved.")

if __name__ == '__main__':
    format_for_fasttext(
        "data/openwebmath_samples.txt",
        "data/fineweb_samples.txt",
        "data/train.txt",
        "data/test.txt"
    )