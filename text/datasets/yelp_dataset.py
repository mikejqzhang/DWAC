import os
import errno
import json

from spacy.lang.en import English
from torchvision.datasets.utils import download_url

from utils import file_handling as fh
from text.datasets.text_dataset import TextDataset, Vocab, tokenize


class YelpDataset(TextDataset):
    """
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, load the training data, otherwise test
        strip_html (bool, optional): If True, remove html tags during preprocessing; default=True
        lower (bool, optional): If true, lowercase text
    """
    url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

    raw_folder = 'raw'
    raw_filename = 'yelp_train.json'
    processed_folder = 'processed'
    train_file = 'yelp_train.jsonlist'
    test_file = 'yelp_test.jsonlist'
    vocab_file = 'yelp_vocab.json'
    classes = ['neg', 'pos']
    class_to_idx = {_class: i for i, _class in enumerate(classes)}

    def __init__(self, root, train=True, strip_html=True, lower=True, is_ood=True):
        super().__init__()
        self.root = os.path.expanduser(root)
        self.train = train
        self.strip_html = strip_html
        self.is_ood = is_ood

        if not self._check_raw_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        self.preprocess()

        if train:
            self.all_docs = fh.read_jsonlist(os.path.join(self.root, self.processed_folder, self.train_file))
        else:
            self.all_docs = fh.read_jsonlist(os.path.join(self.root, self.processed_folder, self.test_file))

        # Do lower-casing on demand, to avoid redoing slow tokenization
        if lower:
            for doc in self.all_docs:
                doc['tokens'] = [token.lower() for token in doc['tokens']]

        # load and build a vocabulary, also lower-casing if necessary

        if not self.is_ood:
            vocab = fh.read_json(os.path.join(self.root, self.processed_folder, self.vocab_file))
            if lower:
                vocab = list(set([token.lower() for token in vocab]))
            self.vocab = Vocab(vocab, add_pad_unk=True)

            self.label_vocab = Vocab(self.classes)



    def _check_processed_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.train_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.vocab_file))

    def _check_raw_exists(self):
        return os.path.exists(os.path.join(self.root, self.raw_folder, self.raw_filename))

    def preprocess(self):
        """Preprocess the raw data file"""
        if self._check_processed_exists():
            return

        try:
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        print("Preprocessing raw data")
        print("Loading spacy")
        # load a spacy parser
        tokenizer = English()

        train_lines = []
        test_lines = []
        vocab = set()

        print("Processing document")
        # read in the raw data
        raw_files = {'train': 'yelp_train.json', 'test': 'yelp_test.json'}
        for split, raw_file in raw_files.items():
            with open(os.path.join(self.root, self.raw_folder, raw_file), 'r') as f:
                for line in f:
                    blob = json.loads(line)
                    text = blob['text']
                    stars = int(blob['stars'])
                    if stars == 3:
                        continue
                    label = 'pos' if stars > 4 else 'neg'
                    text = tokenize(tokenizer, text, strip_html=self.strip_html)
                    # save the text, label, and original file name
                    doc = {'tokens': text.split(), 'label': label, 'rating': stars}
                    if split == 'train':
                        train_lines.append(doc)
                        vocab.update(doc['tokens'])
                    elif split == 'test':
                        test_lines.append(doc)
                    else:
                        raise ValueError("Unexpected split:", split)

        vocab = list(vocab)
        vocab.sort()

        print("Saving processed data")
        fh.write_jsonlist(train_lines, os.path.join(self.root, self.processed_folder, self.train_file))
        fh.write_jsonlist(test_lines, os.path.join(self.root, self.processed_folder, self.test_file))
        if not self.is_ood:
            fh.write_json(vocab, os.path.join(self.root, self.processed_folder, self.vocab_file), sort_keys=False)
