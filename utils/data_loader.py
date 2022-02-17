"""
Data loader for TACRED and NYT.
"""
from collections import Counter
import json
import numpy as np
import os
import random
from utils import constant


def load_hidden(data_dir, model_type, data_type):
    hidden_repr = None
    if model_type == "MultiTextVADTransfer":
        with np.load(os.path.join(data_dir, '{}.npz'.format(data_type))) as data:
            hidden_repr = data['hidden_features']
    return hidden_repr


def load_json(path_to_input):
    with open(path_to_input) as f:
        data = []
        for line in f:
            instance = json.loads(line)
            if type(instance) == dict:
                data.append(instance)
            elif type(instance) == list:
                data = instance
    return data


def extract_rules(instance):
    subjs, subje = instance['subj_start'], instance['subj_end']
    objs, obje = instance['obj_start'], instance['obj_end']
    if subje < objs:
        pos_tokens_between = instance['stanford_pos'][subje:objs + 1]
        pos_tokens_between[0] = 'SUBJ-' + instance['subj_type']
        pos_tokens_between[-1] = 'OBJ-' + instance['obj_type']

        tokens_between = instance['token'][subje:objs + 1]
        tokens_between[0] = 'SUBJ-' + instance['subj_type']
        tokens_between[-1] = 'OBJ-' + instance['obj_type']

        try:
            left_window_1 = 'SUBJ_LEFT_' + instance['stanford_ner'][subjs - 1]
        except IndexError:
            left_window_1 = ''
        try:
            right_window_1 = 'OBJ_RIGHT_' + instance['stanford_ner'][obje + 1]
        except IndexError:
            right_window_1 = ''

        entity_first = 'SUBJ'
    elif obje < subjs:
        pos_tokens_between = instance['stanford_pos'][obje:subjs + 1]
        pos_tokens_between[0] = 'OBJ-' + instance['obj_type']
        pos_tokens_between[-1] = 'SUBJ-' + instance['subj_type']

        tokens_between = instance['token'][obje:subjs + 1]
        tokens_between[0] = 'OBJ-' + instance['obj_type']
        tokens_between[-1] = 'SUBJ-' + instance['subj_type']

        try:
            left_window_1 = 'OBJ_LEFT_' + instance['stanford_ner'][objs - 1]
        except IndexError:
            left_window_1 = ''
        try:
            right_window_1 = 'SUBJ_RIGHT_' + instance['stanford_ner'][subje + 1]
        except IndexError:
            right_window_1 = ''

        entity_first = 'OBJ'
    pos_tokens_between = ' '.join(pos_tokens_between)
    tokens_between = ' '.join(tokens_between)
    return tokens_between, pos_tokens_between, left_window_1, right_window_1, entity_first


def check_rule(instance, rule):
    return rule in extract_rules(instance)


class DataLoader(object):
    """
    Load data from array of dicts, preprocess and prepare batches.
    """

    def __init__(self, data, batch_size, opt, vocab, label2id, hidden_repr, all_rules,
                 evaluation=False, rules=None):
        """Creates a DataLoader object from input data
        args:
            data: list of dicts
        """
        self.batch_size = batch_size
        self.label2id = label2id
        self.all_rules = all_rules
        self.num_classes = len(self.label2id)
        self.opt = opt
        self.vocab = vocab
        self.eval = evaluation

        if hidden_repr is None:
            # create dummy vector for later simplicity
            hidden_repr = np.zeros((len(data), 5))
        data = self.preprocess(data, hidden_repr, vocab, opt)
        # shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        id2label = dict([(v, k) for k, v in self.label2id.items()])
        self.labels = [id2label[np.argmax(d[-1])] for d in data]
        self.num_examples = len(data)
        self.prior = np.sum([d[-1] for d in data], axis=0) / self.num_examples

        # chunk into batches
        if not rules:
            data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
            self.data = data
        else:
            self.data = self.build_predicate_batches(data, rules)
        print("{} batches created from {} instances".format(len(self.data), self.num_examples))

    def preprocess(self, data, hidden_repr, vocab, opt):
        """ Preprocess the data and convert to ids. """
        processed = []
        for d, hidden in zip(data, hidden_repr):
            tokens = d['token']
            if opt['lower']:
                tokens = [t.lower() for t in tokens]
            # anonymize tokens
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']
            tokens[ss:se + 1] = ['SUBJ-' + d['subj_type']] * (se - ss + 1)
            tokens[os:oe + 1] = ['OBJ-' + d['obj_type']] * (oe - os + 1)

            try:
                # Extract dependency path
                path = []
                head = d['stanford_head'][ss]
                path.append(tokens[ss])
                while head != 0:
                    path.append(d['token'][head - 1])
                    head = d['stanford_head'][head - 1]
                head = d['stanford_head'][os]
                path.append(tokens[os])
                while head != 0:
                    path.append(d['token'][head - 1])
                    head = d['stanford_head'][head - 1]
            except KeyError:
                # Extract intertext tokens with a window of size 3
                start = min(0, min(ss, os) - 3)
                end = max(max(se, oe) + 4, len(tokens))
                path = tokens[start:end]

            tokens = map_to_ids(tokens, vocab.word2id)
            path = map_to_ids(path, vocab.word2id)
            # Multihot vector for dependency path
            '''
            path_counter = Counter(path)
            path_multihot = np.zeros(len(vocab.word2id))
            for k, v in path_counter.most_common():
                path_multihot[k] = v
            # Remove PAD token
            path_multihot = path_multihot[1:]
            path_multihot /= np.sum(path_multihot)
            '''
            subj_positions = get_positions(d['subj_start'], d['subj_end'], len(tokens))
            obj_positions = get_positions(d['obj_start'], d['obj_end'], len(tokens))
            relation = np.zeros(self.num_classes)
            if d['relation'] in self.label2id:
                relation[self.label2id[d['relation']]] = 1
            else:
                relation[0] = 1
            processed += [
                (d, tokens, path, subj_positions, obj_positions, hidden, relation)]
        return processed

    def build_predicate_batches(self, data, rules):
        """Build batches by given predicates so that each batch contains only instances that satisfy a predicate.
        :param
            data: a list of instances
            predicates: a dict of predicates predicate -> prior
        """
        rule2instances = {rule: [] for rule in rules}
        rule2instances["no_predicate"] = []
        for instance in data:
            if_p = False
            for rule in rules:
                if check_rule(instance[0], rule):
                    rule2instances[rule].append(instance)
                    if_p = True
                    break
            if not if_p:
                rule2instances["no_predicate"].append(instance)
        batches = []
        for rule, instances in rule2instances.items():
            print(f"{len(instances)} instances satisfy {rule}")
            if not self.eval:
                indices = list(range(len(instances)))
                random.shuffle(indices)
                instances = [instances[i] for i in indices]
            try:
                prior = self.all_rules[rule]
            except KeyError:
                # uniform for binary classification
                prior = [1 / self.num_classes for _ in range(self.num_classes)]
                # prior = [1 - 0.439224, 0.439224]  # posterior label distribution for pattern
                # "title person"
                # prior = [1 - 0.410742, 0.410742]  # posterior label distribution for 4 patterns
                # "person , ... title"
                # prior = [0.48, 0.52]
            batches.extend([(instances[i:i + self.batch_size], prior)
                            for i in range(0, len(instances), self.batch_size)])
            print(f"{len(batches)} batches after {rule}")
        if not self.eval:
            indices = list(range(len(batches)))
            random.shuffle(indices)
            batches = [batches[i] for i in indices]
        # print(len(batches[0]), len(batches[0][0]), len(batches[0][0][0]))
        return batches

    def gold(self):
        """ Return gold labels as a list. """
        return self.labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        prior = [1 / self.num_classes for _ in range(self.num_classes)]

        #  Important for batches with priors ! might be a bug here.
        if len(batch) == 2 and type(batch) == tuple:
            batch, prior = batch

        batch = list(zip(*batch))
        assert len(batch) == 7

        # word dropout
        if not self.eval:
            words = [word_dropout(sent, self.opt['word_dropout']) for sent in batch[1]]
        else:
            words = batch[1]

        # convert to tensors
        hidden = np.array(batch[-2])
        subj_positions = np.array(batch[3]) + constant.MAX_LEN
        obj_positions = np.array(batch[4]) + constant.MAX_LEN
        paths = np.array(batch[2])
        rels = np.array(batch[-1])

        return np.array(words), paths, subj_positions, obj_positions, prior, hidden, rels

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)


def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    if len(ids) < constant.MAX_LEN:
        for _ in range(len(ids), constant.MAX_LEN):
            ids.append(constant.PAD_ID)
    if len(ids) > constant.MAX_LEN:
        ids = ids[:constant.MAX_LEN]
    return np.array(ids)


def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0] * (end_idx - start_idx + 1) + \
           list(range(1, length - end_idx))


def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]


def word_dropout(tokens, dropout):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout
            else x for x in tokens]

