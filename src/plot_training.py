import re
import matplotlib.pyplot as plt

def parse_training_log(log_path):
    """
    解析FastText训练日志文件，提取训练进度、平均损失和学习率。
    """
    pattern = re.compile(
        r'Progress:\s+(\d+\.\d+)%.*lr:\s+([-\d\.]+)\s+avg\.loss:\s+([\d\.]+)'
    )
    progress, avg_loss, lr = [], [], []

    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    progress.append(float(match.group(1)))
                    lr.append(float(match.group(2)))
                    avg_loss.append(float(match.group(3)))
    except FileNotFoundError:
        print(f"文件未找到: {log_path}")
    except Exception as e:
        print(f"解析日志出错: {e}")

    return progress, avg_loss, lr


def plot_training_curve(progress, avg_loss, lr):
    """
    根据训练数据绘制训练进度-损失-学习率曲线图
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_loss = 'tab:blue'
    ax1.set_xlabel('Training Progress (%)')
    ax1.set_ylabel('Average Loss', color=color_loss)
    ax1.plot(progress, avg_loss, color=color_loss, label='Avg Loss')
    ax1.tick_params(axis='y', labelcolor=color_loss)

    ax2 = ax1.twinx()
    color_lr = 'tab:red'
    ax2.set_ylabel('Learning Rate', color=color_lr)
    ax2.plot(progress, lr, color=color_lr, label='Learning Rate')
    ax2.tick_params(axis='y', labelcolor=color_lr)

    plt.title('FastText Training Curve')
    fig.tight_layout()
    plt.show()
    plt.savefig('./log/training_curve.png')


if __name__ == "__main__":
    log_path = './log/training.log'
    progress, avg_loss, lr = parse_training_log(log_path)
    if progress and avg_loss and lr:
        plot_training_curve(progress, avg_loss, lr)
    else:
        print("日志文件中未提取到有效训练数据。")
