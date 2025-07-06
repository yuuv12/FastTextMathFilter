import fasttext
import argparse

def train(input, output, lr, epoch, wordNgrams, dim):
    model = fasttext.train_supervised(
        input=input,
        lr=lr,
        epoch=epoch,
        wordNgrams=wordNgrams,
        dim=dim,
        loss='softmax'
    )

    model.save_model(output)
    print(f"Model trained and saved to {output}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/train.txt')
    parser.add_argument('--output', type=str, default='model.bin')
    parser.add_argument('--lr', type=float, default=0.5)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--wordNgrams', type=int, default=2)
    parser.add_argument('--dim', type=int, default=100)
    args = parser.parse_args()
    
    train(args.input, args.output, args.lr, args.epoch, args.wordNgrams, args.dim)

if __name__ == '__main__':
    main()
