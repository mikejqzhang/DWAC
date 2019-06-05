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
from text.run import load_embeddings, load_data
 
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
    #parser.add_argument('--glove-file', type=str, default='/cse/web/courses/cse447/19wi/assignments/resources/glove/glove.6B.300d.txt.gz',
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

    print("Start predicting")
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
    # create the model
    model_file = os.path.join(args.output_dir, 'model.best.tar')
    print("Reloading best model")
    model.load(model_file)



    train(args, label_vocab, model, train_loader, dev_loader, test_loader, ref_loader, ood_loader)

def train(args, label_vocab, model, train_loader, dev_loader, test_loader, ref_loader, ood_loader=None):
    best_dev_acc = 0.0
    done = False
    epoch = 0
    epochs_without_improvement = 0
    best_epoch = 0

    print("Creating output directory {:s}".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    print("Doing test eval")
    if args.model == 'dwac':
        test_output = test_fast(args, model, test_loader, ref_loader, name='Test')
        test_acc = test_output['accuracy']
        save_output(os.path.join(args.output_dir, 'test.npz'), test_output)
    else:
        test_acc, test_labels, test_indices, test_pred_probs, test_z, test_confs, test_atts = test(
            args, label_vocab, model, test_loader, ref_loader, name='Test')
        print("Saving")



def test(args, label_vocab, model, test_loader, ref_loader, name='Test', return_acc=False):
    test_loss = 0
    correct = 0
    true_labels = []
    all_indices = []
    pred_probs = []
    confs = []
    zs = []
    atts = []
    n_items = 0
    with open(os.path.join(args.output_dir, 'predicted.tsv'), 'w+') as predicted, open(os.path.join(args.output_dir, 'actual.tsv'), 'w+') as actual:
        for batch_idx, (data, target, indices) in enumerate(test_loader):
            data, target = data.to(args.device), target.to(args.device)
            output = model.evaluate(data, target, ref_loader)
            test_loss += output['total_loss'].item()
            batch_size = len(target)
            n_items += batch_size
            pred = output['probs'].max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            for t, p in zip(target, pred):
                actual.write(label_vocab.idx2word[t.item()] + "\n")
                predicted.write(label_vocab.idx2word[p.item()] + "\n")
            all_indices.extend(list(to_numpy(indices, args.device)))
            if not return_acc:
                true_labels.extend(list(to_numpy(target, args.device)))
                pred_probs.append(to_numpy(output['probs'].exp(), args.device))
                # all_indices.extend(list(to_numpy(indices, args.device)))
                zs.append(to_numpy(output['z'], args.device))
                atts.append(to_numpy(output['att'], args.device))
                if args.model == 'dwac':
                    confs.append(to_numpy(output['confs'], args))

    test_loss /= len(test_loader.sampler)
    print('{:s} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        name, test_loss, correct, len(test_loader.sampler),
        100. * correct / len(test_loader.sampler)))
    print()

    acc = correct / len(test_loader.sampler)
    if args.model == 'dwac' and not return_acc:
        confs = np.vstack(confs)

    if return_acc:
        return acc, all_indices
    else:
        max_att_len = np.max([m.shape[1] for m in atts])
        att_matrix = np.zeros([n_items, max_att_len])
        index = 0
        for m in atts:
            batch_size, width = m.shape
            att_matrix[index:index+batch_size, :width] = m.copy()
            index += batch_size

        return (acc, true_labels, all_indices,
                np.vstack(pred_probs), np.vstack(zs), confs, att_matrix)

if __name__ == "__main__":
    main()
