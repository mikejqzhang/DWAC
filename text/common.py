import os

from text.datasets.imdb_dataset import IMDB
from text.datasets.amazon_dataset import AmazonReviews
from text.datasets.stackoverflow_dataset import StackOverflowDataset
from text.datasets.subjectivity_dataset import SubjectivityDataset
from text.datasets.yelp_dataset import YelpDataset

def load_dataset(root_dir, dataset, subset=None, lower=False, ood_class=None):

    ood_dataset = None
    if dataset == 'imdb':
        train_dataset = IMDB(os.path.join(root_dir, 'imdb'), partition='train', download=True, strip_html=True, lower=lower)
        test_dataset = IMDB(os.path.join(root_dir, 'imdb'), partition='test', download=True, strip_html=True, lower=lower)
        if ood_class:
            ood_dataset = IMDB(os.path.join(root_dir, 'imdb'), partition='ood', download=False, strip_html=True, lower=lower, ood_class='yelp')
    elif dataset == 'amazon':
        if subset is None:
            raise ValueError("Please provide a subset for the Amazon dataset.")
        train_dataset = AmazonReviews(os.path.join(root_dir, 'amazon'), subset=subset, train=True, download=True, lower=lower)
        test_dataset = AmazonReviews(os.path.join(root_dir, 'amazon'), subset=subset, train=False, download=True, lower=lower)
    elif dataset == 'stackoverflow':
        train_dataset = StackOverflowDataset(os.path.join(root_dir, 'stackoverflow'),
                                             partition='train', download=True, lower=lower,
                                             process=True,
                                             ood_class=ood_class)
        test_dataset = StackOverflowDataset(os.path.join(root_dir, 'stackoverflow'),
                                            partition='test', download=True, lower=lower,
                                            process=False,
                                            ood_class=ood_class)
        ood_dataset = StackOverflowDataset(os.path.join(root_dir, 'stackoverflow'),
                                           partition='ood', download=True, lower=lower,
                                           process=False,
                                           ood_class=ood_class)
    elif dataset == 'subjectivity':
        train_dataset = SubjectivityDataset(os.path.join(root_dir, 'subjectivity'), train=True, download=True, lower=lower)
        test_dataset = SubjectivityDataset(os.path.join(root_dir, 'subjectivity'), train=False, download=True, lower=lower)
    elif dataset == 'yelp':
        train_dataset = YelpDataset(os.path.join(root_dir, 'yelp'), train=True, lower=lower)
        test_dataset = YelpDataset(os.path.join(root_dir, 'yelp'), train=False, lower=lower)
    else:
        raise ValueError("Dataset not recognized.")

    return train_dataset, test_dataset, ood_dataset
