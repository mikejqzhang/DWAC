import os
import json
import gzip
import argparse

import numpy as np

import torch
from torch.utils.data.sampler import SubsetRandomSampler

from text.datasets.text_dataset import collate_fn
from text.baseline_model import TextBaseline
from text.dwac_model import AttentionCnnDwac
from text.proto_model import ProtoDwac
from text.common import load_dataset
from utils.common import to_numpy

def main():
    parser = argparse.ArgumentParser(description='Text Classifier',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Fundamental options
    parser.add_argument('--model', type=str, default='baseline', metavar='N',
                        help='Model to use [baseline|dwac]')
    parser.add_argument('--dataset', type=str, default='imdb', metavar='N',
                        help='Dataset to run [imdb|amazon|stackoverflow|subjectivity]')
    parser.add_argument('--subset', type=str, default=None, metavar='N',
                        help='Subset for amazon or framing dataset [beauty|...]')

    # Text Options
    parser.add_argument('--lower', action='store_true', default=False,
                        help='Convert text to lower case')

    # Model Options
    parser.add_argument('--glove-file', type=str, default='data/vectors/glove.6B.300d.txt.gz',
                        metavar='N', help='Glove vectors')
    parser.add_argument('--embedding-dim', type=int, default=300, metavar='N',
                        help='word vector dimensions')
    parser.add_argument('--hidden-dim', type=int, default=100, metavar='N',
                        help='Size of penultimate layer')
    parser.add_argument('--fix-embeddings', action='store_true', default=False,
                        help='fix word embeddings in training')
    parser.add_argument('--kernel-size', type=int, default=5, metavar='N',
                        help='convolution filter kernel size')

    # File Options
    parser.add_argument('--root-dir', type=str, default='./data', metavar='PATH',
                        help='path to data directory')
    parser.add_argument('--output-dir', type=str, default='./data/temp', metavar='PATH',
                        help='Output directory')

    # Training Options
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                        help='test batch size')
    parser.add_argument('--dev-prop', type=float, default=0.1, metavar='N',
                        help='Proportion of training data to use as a dev set')
    parser.add_argument('--lr', type=float, default=0.001, metavar='N',
                        help='learning rate for training')
    parser.add_argument('--max-epochs', type=int, default=50, metavar='N',
                        help='number of training epochs')
    parser.add_argument('--patience', type=int, default=5, metavar='N',
                        help='number of training epochs')

    # DWAC Architecture Options
    parser.add_argument('--z-dim', type=int, default=5, metavar='N',
                        help='dimensions of latent representation')
    parser.add_argument('--kernel', type=str, default='gaussian', metavar='k',
                        help='type of distance function [gaussian|laplace|invquad')
    parser.add_argument('--gamma', type=float, default=1, metavar='k',
                        help='hyperparameter for kernel')
    parser.add_argument('--eps', type=float, default=1e-12, metavar='k',
                        help='label smoothing factor for learning')
    parser.add_argument('--topk', type=int, default=10, metavar='N',
                        help='top k nearest neighbors to compare to at test time')

    # ProtoDWAC Architecture Options
    parser.add_argument('--n-proto', type=int, default=5, metavar='N',
                        help='number of prototypes per class')

    # Running Options
    parser.add_argument('--device', type=int, default=None,
                        help='GPU to use (if any)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--seed', type=int, default=None, metavar='N',
                        help='random seed')

    args = parser.parse_args()

    if args.device is None:
        args.device = 'cpu'
    else:
        args.device = 'cuda:' + str(args.device)
    print("Using device:", args.device)
    args.update_embeddings = not args.fix_embeddings

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.device != 'cpu':
            torch.backends.cudnn.deterministic = True
            torch.cuda.manual_seed_all(args.seed)

    # load data and create vocab and label vocab objects
    vocab, label_vocab, train_loader, dev_loader, test_loader, ref_loader, ood_loader = \
        load_data(args)
    args.n_classes = len(label_vocab)

    # load an initialize the embeddings
    embeddings_matrix = load_embeddings(args, vocab)

    # create the model
    if args.model == 'baseline':
        print("Creating baseline model")
        model = TextBaseline(args, vocab, embeddings_matrix)
    elif args.model == 'dwac':
        print("Creating DWAC model")
        model = AttentionCnnDwac(args, vocab, embeddings_matrix)
    elif args.model == 'proto':
        print("Creating Prototyped DWAC model")
        model = ProtoDwac(args, vocab, embeddings_matrix)
    else:
        raise ValueError("Model type not recognized.")
    print("Update embeddings = ", args.update_embeddings)

    train(args, model, train_loader, dev_loader, test_loader, ref_loader, ood_loader)


def train(args, model, train_loader, dev_loader, test_loader, ref_loader, ood_loader=None):
    best_dev_acc = 0.0
    done = False
    epoch = 0
    epochs_without_improvement = 0
    best_epoch = 0

    print("Creating output directory {:s}".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model_file = os.path.join(args.output_dir, 'model.best.tar')
    print("Reloading best model")
    model.load(model_file)

    print("Doing test eval")
    if args.model == 'dwac':
        test_output = test_fast(args, model, test_loader, ref_loader, name='Test')
        test_acc = test_output['accuracy']
        save_output(os.path.join(args.output_dir, 'test.npz'), test_output)
    else:
        test_acc, test_labels, test_indices, test_pred_probs, test_z, test_confs, test_atts = test(
            args, model, test_loader, ref_loader, name='Test')
        print("Saving")
        np.savez(os.path.join(args.output_dir, 'test.npz'),
                 labels=test_labels,
                 z=test_z,
                 pred_probs=test_pred_probs,
                 indices=test_indices,
                 confs=test_confs,
                 atts=test_atts)


    print('Saving Dev+Test Metrics')
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as metrics_f:
        json.dump({'dev_acc': dev_acc, 'test_acc': test_acc, 'best_epoch': best_epoch},
                  metrics_f,
                  indent=4)
