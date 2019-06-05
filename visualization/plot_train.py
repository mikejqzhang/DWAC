import matplotlib as mpl
mpl.use('Agg')

import os
from optparse import OptionParser

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

stack_overflow_classes = ['wordpress',
               'oracle',
               'svn',
               'apache',
               'excel',
               'matlab',
               'visual-studio',
               'cocoa',
               'osx',
               'bash',
               'spring',
               'hibernate',
               'scala',
               'sharepoint',
               'ajax',
               'qt',
               'drupal',
               'linq',
               'haskell',
               'magento']

imdb_classes = ['Negative', 'Postive']
plot_function = TSNE

def main():
    plt.rcParams["figure.figsize"] = (10,10)
    usage = "%prog exp_dir"
    parser = OptionParser(usage=usage)
    parser.add_option('--split', action="store_true", dest="split", default=False,
                      help='Split into multiple plots: default=%default')
    parser.add_option('--seed', dest='seed', default=42,
                      help='Random seed: default=%default')

    (options, args) = parser.parse_args()
    indir = args[0]
    dataset = args[1]
    num_proto = args[2]

    class_list = stack_overflow_classes if dataset == 'stack_overflow' else imdb_classes

    train_file = os.path.join(indir, 'train.npz')
    dev_file = os.path.join(indir, 'dev.npz')

    seed = int(options.seed)
    split = options.split
    np.random.seed(seed)

    train_data = np.load(train_file)
    dev_data = np.load(dev_file)
    viz_data(train_data, 'train', dataset,  num_proto, class_list)
    viz_data(dev_data, 'dev', dataset, num_proto, class_list)

def viz_data(data, name, dataset, num_proto, class_list, split=False):
    z = data['z']
    labels = data['labels']
    n, dz = z.shape
    print(z.shape)

    n_classes = np.max(labels+1)

    # scatter the labels
    labels = scatter(labels, n_classes)
    plotter = plot_function(n_components=2)
    reduced_z = plotter.fit_transform(z)

    if split:
        fig, axes = plt.subplots(nrows=1, ncols=n_classes, figsize=(n_classes*2, 2), sharex=True, sharey=True)
    else:
        fig, ax = plt.subplots()
    scatters = []
    for k in range(n_classes):
        indices = np.array(labels[:, k], dtype=bool)
        if split:
            axes[k].scatter(z[indices, 0], z[indices, 1], c='k', alpha=0.5)
        else:
            scatters.append(ax.scatter(z[indices, 0], z[indices, 1], s=1, alpha=0.9))

    ax.legend(scatters, class_list)
    plt.title('t-SNE of {} with {} prototypes'.format(dataset, num_proto))
    plt.savefig('tSNE_{}_{}_{}.pdf'.format(dataset, num_proto, name))


def scatter(labels, n_classes):
    if len(labels.shape) == 1 or labels.shape[1] == 1:
        n_items = len(labels)
        temp = np.zeros((n_items, n_classes), dtype=int)
        temp[np.arange(n_items), labels] = 1
        labels = temp
    return labels


if __name__ == '__main__':
    main()
