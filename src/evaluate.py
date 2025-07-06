import fasttext
from sklearn.metrics import classification_report, confusion_matrix
import argparse

def load_fasttext_labels(file_path):
    labels = []
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            label, text = line.strip().split(' ', 1)
            labels.append(label)
            texts.append(text)
    return texts, labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model.bin')
    parser.add_argument('--test', type=str, default='data/test.txt')
    args = parser.parse_args()

    model = fasttext.load_model(args.model)
    texts, y_true = load_fasttext_labels(args.test)

    y_pred = []
    for text in texts:
        label = model.predict(text)[0][0]
        y_pred.append(label)

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["math", "other"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=["__label__math", "__label__other"]))

if __name__ == '__main__':
    main()
