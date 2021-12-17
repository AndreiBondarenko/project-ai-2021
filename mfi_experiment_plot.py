import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import os
from rich.console import Console
from rich.table import Table


def parser():
    _parser = argparse.ArgumentParser(description='Plot Matrix Factorization with Implicit Feedback Experiment Results.')
    _parser.add_argument('dir', metavar='dir_name', help='name of the experiment\'s output folder')
    return _parser


if __name__ == '__main__':
    args = parser().parse_args()

    if not os.path.isdir(f'./out/{args.dir}'):
        print('Experiment with given doesn\'t exist.')

    with open(f'./out/{args.dir}/losses.pkl', 'rb') as f:
        losses = pickle.load(f)

    with open(f'./out/{args.dir}/metrics.pkl', 'rb') as f:
        metrics = pickle.load(f)

    train_losses, test_losses = losses
    r5, r10, ndcg5, ndcg10 = metrics

    for i, (train, test) in enumerate(zip(train_losses, test_losses)):
        if i == 0:
            plt.plot(train, alpha=0.3, color='#1f77b4', label='train')
            plt.plot(test, alpha=0.3, color='#ff7f0e', label='test')
        else:
            plt.plot(train, alpha=0.3, color='#1f77b4')
            plt.plot(test, alpha=0.3, color='#ff7f0e')
        plt.xlabel('ALS iteration')
        plt.ylabel('Loss')
        plt.legend()
    # plt.show()

    console = Console()

    table = Table(show_header=True, header_style='bold magenta')
    table.add_column('Fold')
    table.add_column('Recall@5')
    table.add_column('Recall@10')
    table.add_column('nDCG@5')
    table.add_column('nDCG@10')
    avg_r5 = [np.mean(r) for r in r5]
    avg_r10 = [np.mean(r) for r in r10]
    avg_n5 = [np.mean(r) for r in ndcg5]
    avg_n10 = [np.mean(r) for r in ndcg10]
    for i, (r1, r2, n1, n2) in enumerate(zip(avg_r5, avg_r10, avg_n5, avg_n10), start=1):
        table.add_row(
            f'{i}',
            f'{r1:.5f}',
            f'{r2:.5f}',
            f'{n1:.5f}',
            f'{n2:.5f}',
            end_section=i == 5
        )
    table.add_row(
            f'',
            f'{np.mean(avg_r5):.5f} \u00B1 {np.std(avg_r5):.5f}',
            f'{np.mean(avg_r10):.5f} \u00B1 {np.std(avg_r10):.5f}',
            f'{np.mean(avg_n5):.5f} \u00B1 {np.std(avg_n5):.5f}',
            f'{np.mean(avg_n10):.5f} \u00B1 {np.std(avg_n10):.5f}',
    )

    console.print(table)






