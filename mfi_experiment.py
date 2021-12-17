from tqdm.auto import tqdm
import scipy.sparse as sps
import argparse
import pickle
import os

from recommenders.mfi import MatrixFactorizationImplicit
from recommenders.mfi import utils as mfi_utils
import utils


def parser():
    _parser = argparse.ArgumentParser(description='Run Matrix Factorization with Implicit Feedback Experiment.')
    _parser.add_argument('dir', metavar='dir_name', help='name of the experiment\'s output folder')
    _parser.add_argument('k', metavar='K', help='number of latent factors', type=int)
    _parser.add_argument('reg', metavar='\u03BB', help='regularization factor', type=float)
    return _parser


if __name__ == '__main__':
    args = parser().parse_args()

    try:
        os.mkdir(f'./out/{args.dir}')
    except FileExistsError:
        print('Experiment with given name already exists, delete it or pick another name.')

    # init train-test folds and recommenders to be trained
    train_sets = [sps.load_npz(f'./data/train{i}.npz') for i in range(1, 6)]
    test_sets = [sps.load_npz(f'./data/test{i}.npz') for i in range(1, 6)]
    recommenders = [MatrixFactorizationImplicit(args.k, args.reg) for _ in range(5)]

    # init lists for losses
    train_losses = []
    test_losses = []

    # train on each fold
    for rec, train, test in zip(recommenders, train_sets, test_sets):
        train_loss, test_loss = rec.train(train, test)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

    # pickle losses
    with open(f'./out/{args.dir}/losses.pkl', 'wb') as f:
        pickle.dump((train_losses, test_losses), f)

    # pickle models
    for i, rec in enumerate(recommenders, start=1):
        rec.save(f'./out/{args.dir}/model-fold{i}.pkl')

    # compute metrics
    rec_at_5 = []
    rec_at_10 = []
    ndcg_at_5 = []
    ndcg_at_10 = []

    for rec, test in zip(recommenders, test_sets):
        _test = mfi_utils.build_conf_mat(test).tocsr()
        rec_at_5.append([])
        rec_at_10.append([])
        ndcg_at_5.append([])
        ndcg_at_10.append([])
        for i in tqdm(range(_test.shape[0])):
            topk = rec.recommend(k=10, user=i)
            actual = _test[i]
            rec_at_5[-1].append(utils.recall_at_k(topk[:5], actual))
            rec_at_10[-1].append(utils.recall_at_k(topk, actual))
            ndcg_at_5[-1].append(utils.ndcg_at_k(topk[:5], actual))
            ndcg_at_10[-1].append(utils.ndcg_at_k(topk, actual))

    with open(f'./out/{args.dir}/metrics.pkl', 'wb') as f:
        pickle.dump((rec_at_5, rec_at_10, ndcg_at_5, ndcg_at_10), f)




