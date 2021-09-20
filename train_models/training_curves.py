import os
import glob
import re
from io import StringIO
import numpy as np
import matplotlib
from matplotlib import pyplot as plt


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def parse_values(log_bytes, max_epochs=None):
    train_curve = []
    val_curve = []
    for line in log_bytes.readlines():
        if not line.startswith('EPOCH'):
            continue
        split_line = line.split()
        epoch = int(split_line[1])
        acc = float(split_line[5])
        loss = float(split_line[9])
        subset = split_line[3]
        curve = train_curve if 'train' in subset.lower() else val_curve
        if max_epochs and len(curve) >= max_epochs:
            continue
        curve.append((epoch, acc, loss))
    return train_curve, val_curve


def smooth_curves(*curves, window=8):
    return_curves = []
    for curve in curves:
        epochs, acc, loss = zip(*curve)
        smooth_acc = moving_average(acc, window)
        smooth_loss = moving_average(loss, window)
        return_curves.append(list(zip(epochs, smooth_acc, smooth_loss)))
    return return_curves


def write_stats(train_curve, val_curve, train_stats, output_path):
    train_epochs, train_acc, train_loss = zip(*train_curve)
    val_epochs, val_acc, val_loss = zip(*val_curve)

    min_pos = np.argmin(val_loss)
    train_stats.write('{:60s}|{:12d}|{:12.4f}|{:12.4f}|\n'.format(os.path.basename(output_path), val_epochs[min_pos],
                                                                  val_acc[min_pos], train_acc[min_pos]))
    # print(os.path.basename(output_path), val_epochs[min_pos], val_acc[min_pos], train_acc[min_pos])


def plot_training_curves(train_curve, val_curve, output_path):
    train_epochs, train_acc, train_loss = zip(*train_curve)
    val_epochs, val_acc, val_loss = zip(*val_curve)

    fig = plt.figure(figsize=(7, 5))
    ax1 = fig.gca()

    train_color = 'tab:red'
    val_color = 'tab:blue'

    ax1.plot(train_epochs, train_acc, color=train_color, label='train accuracy')
    ax1.plot(val_epochs, val_acc, color=val_color, label='validation accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_yticks([x / 10 for x in range(11)])
    ax1.tick_params(axis='y')
    ax1.set_ylim(0, 1.0)

    ax1.set_xlim(0, len(train_epochs))
    if len(train_epochs) < 20:
        ax1.set_xticks([x + 1 for x in range(len(train_epochs))])
    else:
        ax1.set_xticks([int(x) for x in np.linspace(0, len(train_epochs), 10)])
    ax1.yaxis.grid(True, which='major')

    ax2 = ax1.twinx()
    ax2.plot(train_epochs, train_loss, '--', color=train_color, label="train loss")
    ax2.plot(val_epochs, val_loss, '--', color=val_color, label='validation loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_ylim(0, 3.5)
    # ax2.set_yticks([x/2 for x in range(10)])
    # ax2.tick_params(axis='y')

    l = ax1.get_ylim()
    l2 = ax2.get_ylim()
    f = lambda x: l2[0] + (x - l[0]) / (l[1] - l[0]) * (l2[1] - l2[0])
    ticks = f(ax1.get_yticks())
    ax2.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))

    ax1.legend(loc=2)
    ax2.legend(loc=3)

    plt.margins(0.0)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # title = os.path.splitext(os.path.basename(output_path))[0]
    # ax1.set_title(title)
    # print(title, val_epochs[np.argmin(val_loss)])
    # plt.show()


def plot_all_logs(logs_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    clean_file_names = []
    for file_name in os.listdir(logs_folder):
        if not os.path.isfile(os.path.join(logs_folder, file_name)):
            continue
        clean_file_names.append(re.sub(r'_[0-9]*.out', '', file_name))

    unique_file_names = sorted(set(clean_file_names))

    train_stats = open(os.path.join(output_folder, 'train_stats.txt'), 'w')
    train_stats.write('{:^60s}|{:^12s}|{:^12s}|{:^12s}|\n'.format('NAME', 'VAL-EPOCHS', 'VAL-ACC', 'TRAIN-ACC'))
    train_stats.write('-' * 100 + '\n')
    train_stats_smooth = open(os.path.join(output_folder, 'train_stats_smooth.txt'), 'w')
    train_stats_smooth.write('{:^60s}|{:^12s}|{:^12s}|{:^12s}|\n'.format('NAME', 'VAL-EPOCHS', 'VAL-ACC', 'TRAIN-ACC'))
    train_stats_smooth.write('-' * 100 + '\n')
    for u_file_name in unique_file_names:
        image_name = u_file_name + '.jpg'
        output_path = os.path.join(output_folder, image_name)
        matching_logs = glob.glob(os.path.join(logs_folder, u_file_name + '_[0-9]*.out'))
        log_bytes = StringIO()
        for logfile in sorted(matching_logs):
            with open(logfile, 'r') as fin:
                log_bytes.write(fin.read())
        log_bytes.seek(0)
        try:
            train_curve, val_curve = parse_values(log_bytes)
            s_train_curve, s_val_curve = smooth_curves(train_curve, val_curve)
            write_stats(train_curve, val_curve, train_stats, output_path)
            write_stats(s_train_curve, s_val_curve, train_stats_smooth, output_path)
            plot_training_curves(s_train_curve, s_val_curve, output_path)
        except:
            print('{} error'.format(image_name))
    train_stats.close()
    train_stats_smooth.close()


if __name__ == "__main__":
    logs_folder = os.path.abspath(os.path.join(__file__, os.pardir, 'training_logs'))
    output_folder = os.path.abspath(os.path.join(__file__, os.pardir, 'train_curve_plots'))
    plot_all_logs(logs_folder, output_folder)
