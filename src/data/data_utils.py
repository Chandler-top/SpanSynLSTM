from typing import List, Dict, Tuple
from src.data import Instance
import numpy as np


B_PREF="B-"
I_PREF = "I-"
S_PREF = "S-"
E_PREF = "E-"
O = "O"

START_TAG = "<START>"
STOP_TAG = "<STOP>"
PAD = "<PAD>"
UNK = "<UNK>"
root_dep_label = "root"
self_label = "self"

import logging

logger = logging.getLogger(__name__)

def convert_iobes(labels: List[str]) -> List[str]:
	"""
	Use IOBES tagging schema to replace the IOB tagging schema in the instance
	:param insts:
	:return:
	"""
	for pos in range(len(labels)):
		curr_entity = labels[pos]
		if pos == len(labels) - 1:
			if curr_entity.startswith(B_PREF):
				labels[pos] = curr_entity.replace(B_PREF, S_PREF)
			elif curr_entity.startswith(I_PREF):
				labels[pos] = curr_entity.replace(I_PREF, E_PREF)
		else:
			next_entity = labels[pos + 1]
			if curr_entity.startswith(B_PREF):
				if next_entity.startswith(O) or next_entity.startswith(B_PREF):
					labels[pos] = curr_entity.replace(B_PREF, S_PREF)
			elif curr_entity.startswith(I_PREF):
				if next_entity.startswith(O) or next_entity.startswith(B_PREF):
					labels[pos] = curr_entity.replace(I_PREF, E_PREF)
	return labels


def build_label_idx(insts: List[Instance]) -> Tuple[List[str], Dict[str, int]]:
	"""
	Build the mapping from label to index and index to labels.
	:param insts: list of instances.
	:return:
	"""
	label2idx = {}
	idx2labels = []
	label2idx[PAD] = len(label2idx)
	idx2labels.append(PAD)
	for inst in insts:
		for label in inst.labels:
			if label not in label2idx:
				idx2labels.append(label)
				label2idx[label] = len(label2idx)

	label2idx[START_TAG] = len(label2idx)
	idx2labels.append(START_TAG)
	label2idx[STOP_TAG] = len(label2idx)
	idx2labels.append(STOP_TAG)
	label_size = len(label2idx)
	logger.info("#labels: {}".format(label_size))
	logger.info("label 2idx: {}".format(label2idx))
	return idx2labels, label2idx

def build_spanlabel_idx(insts: List[Instance]) -> Tuple[List[str], Dict[str, int]]:
	"""
	Build the mapping from label to index and index to labels.
	:param insts: list of instances.
	:return:
	"""
	label2idx = {}
	idx2labels = []
	label2idx['O'] = len(label2idx)
	idx2labels.append('O')
	for inst in insts:
		for spanlabel in inst.span_labels:
			entity_type = spanlabel[1]
			if entity_type not in label2idx:
				idx2labels.append(entity_type)
				label2idx[entity_type] = len(label2idx)
			else:
				continue

	label_size = len(label2idx)
	print("#span labels: {}".format(label_size))
	print("spanlabel 2idx: {}".format(label2idx))
	return idx2labels, label2idx

def check_all_labels_in_dict(insts: List[Instance], label2idx: Dict[str, int]):
	for inst in insts:
		for label in inst.labels:
			if label not in label2idx:
				raise ValueError(f"The label {label} does not exist in label2idx dict. The label might not appear in the training set.")


def build_word_idx(trains:List[Instance], devs:List[Instance], tests:List[Instance]) -> Tuple[Dict, List, Dict, List]:
	"""
	Build the vocab 2 idx for all instances
	:param train_insts:
	:param dev_insts:
	:param test_insts:
	:return:
	"""
	word2idx = dict()
	idx2word = []
	word2idx[PAD] = 0
	idx2word.append(PAD)
	word2idx[UNK] = 1
	idx2word.append(UNK)

	char2idx = {}
	idx2char = []
	char2idx[PAD] = 0
	idx2char.append(PAD)
	char2idx[UNK] = 1
	idx2char.append(UNK)

	# extract char on train, dev, test
	for inst in trains + devs + tests:
		for word in inst.words:
			if word not in word2idx:
				word2idx[word] = len(word2idx)
				idx2word.append(word)
	# extract char only on train (doesn't matter for dev and test)
	for inst in trains:
		for word in inst.words:
			for c in word:
				if c not in char2idx:
					char2idx[c] = len(idx2char)
					idx2char.append(c)
	return word2idx, idx2word, char2idx, idx2char


def check_all_obj_is_None(objs):
	for obj in objs:
		if obj is not None:
			return False
	return [None] * len(objs)

def build_deplabel_idx(insts: List[Instance]) -> Tuple[Dict[str, int], int]:
	deplabel2idx = {}
	deplabels = []
	if self_label not in deplabel2idx:
		deplabels.append(self_label)
		deplabel2idx[self_label] = len(deplabel2idx)
	for inst in insts:
		for label in inst.dep_labels:
			if label not in deplabels:
				deplabels.append(label)
				deplabel2idx[label] = len(deplabel2idx)
	root_dep_label_id = deplabel2idx[root_dep_label]
	print("dep labels: {}".format(len(deplabels)))
	print("dep label 2idx: {}".format(deplabel2idx))
	return deplabel2idx, root_dep_label_id

def head_to_adj(max_len, words, heads):
    """
    Convert a tree object to an (numpy) adjacency matrix.
    """
    directed = 0
    self_loop = False #config.adj_self_loop
    ret = np.zeros((max_len, max_len), dtype=np.float32)

    for i, head in enumerate(heads):
        if head == 0:
            continue
        ret[head, i] = 1

    if not directed:
        ret = ret + ret.T

    if self_loop:
        for i in range(len(words)):
            ret[i, i] = 1

    return ret


def head_to_adj_label(max_len, words, heads, dep_labels, root_dep_label_id):
    """
    Convert a tree object to an (numpy) adjacency matrix.
    """
    directed = 0
    self_loop = False

    dep_label_ret = np.zeros((max_len, max_len), dtype=np.long)

    for i, head in enumerate(heads):
        if head == 0:
            continue
        dep_label_ret[head, i] = dep_labels[i]

    if not directed:
        dep_label_ret = dep_label_ret + dep_label_ret.T

    if self_loop:
        for i in range(len(words)):
            dep_label_ret[i, i] = root_dep_label_id

    return dep_label_ret