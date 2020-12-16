import torch
import nltk
import itertools
import random
import json
import unicodedata

from .config import *

class Voc:
    def __init__(self, name, word2index=None, index2word=None):
        self.name = name
        self.trimmed = False if not word2index else True # if a precomputed vocab is specified assume the user wants to use it as-is
        self.word2index = word2index if word2index else {"UNK": UNK_token}
        self.word2count = {}
        self.index2word = index2word if index2word else {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: "UNK"}
        self.num_words = 4 if not index2word else len(index2word)  # Count SOS, EOS, PAD, UNK

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {"UNK": UNK_token}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: "UNK"}
        self.num_words = 4 # Count default tokens

        for word in keep_words:
            self.addWord(word)

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Tokenize the string using NLTK
def tokenize(text):
    tokenizer = nltk.tokenize.RegexpTokenizer(pattern=r'\w+|[^\w\s]')
    # simplify the problem space by considering only ASCII data
    cleaned_text = unicodeToAscii(text.lower())

    # if the resulting string is empty, nothing else to do
    if not cleaned_text.strip():
        return []
    
    return tokenizer.tokenize(cleaned_text)

# Create a Voc object from precomputed data structures
def loadPrecomputedVoc(corpus_name, word2index_path, index2word_path):
    with open(word2index_path) as fp:
        word2index = json.load(fp)
    with open(index2word_path) as fp:
        index2word = json.load(fp)
    return Voc(corpus_name, word2index, index2word)

# Given a dialog entry, consisting of a list of {text, label} objects, preprocess
# each utterance's text by tokenizing and truncating.
# Returns the processed dialog entry where text has been replaced with a list of
# tokens, each no longer than MAX_LENGTH - 1 (to leave space for the EOS token)
def processDialog(voc, dialog):
    processed = []
    for utterance in dialog:
        tokens = tokenize(utterance["text"])
        if len(tokens) >= MAX_LENGTH:
            tokens = tokens[:(MAX_LENGTH-1)]
        # replace out-of-vocabulary tokens
        for i in range(len(tokens)):
            if tokens[i] not in voc.word2index:
                tokens[i] = "UNK"
        processed.append({"tokens": tokens, "is_attack": utterance.get("labels", {}).get("is_attack", None), "convo_id": utterance.get("labels", {}).get("id", None)})
    return processed

# Load context-reply pairs from the given dataset.
# Since the dataset may be large, we avoid keeping more data in memory than
# absolutely necessary by cleaning each utterance (tokenize, truncate, replace OOV tokens)
# line by line in this function.
# Returns a list of pairs in the format (context, reply, label)
def loadPairs(voc, path, last_only=False):
    pairs = []
    with open(path) as datafile:
        for i, line in enumerate(datafile):
            print("\rLine {}".format(i+1), end='')
            raw_convo_data = json.loads(line)
            dialog = processDialog(voc, raw_convo_data)
            iter_range = range(1, len(dialog)) if not last_only else [len(dialog)-1]
            for idx in iter_range:
                reply = dialog[idx]["tokens"][:(MAX_LENGTH-1)]
                label = dialog[idx]["is_attack"]
                convo_id = dialog[idx]["convo_id"]
                # gather as context up to CONTEXT_SIZE utterances preceding the reply
                start = max(idx - CONTEXT_SIZE, 0)
                context = [u["tokens"] for u in dialog[start:idx]]
                pairs.append((context, reply, label, convo_id))
        print()
    return pairs

# Using the functions defined above, return a list of pairs for unlabeled training
def loadUnlabeledData(voc, train_path):
    print("Start preparing training data ...")
    print("Preprocessing training corpus...")
    train_pairs = loadPairs(voc, train_path)
    print("Loaded {} pairs".format(len(train_pairs)))
    return train_pairs

# Using the functions defined above, return a list of pairs for labeled training
def loadLabeledData(voc, attack_train_path, attack_val_path, analysis_path):
    print("Start preparing training data ...")
    print("Preprocessing labeled training corpus...")
    attack_train_pairs = loadPairs(voc, attack_train_path, last_only=True)
    print("Loaded {} pairs".format(len(attack_train_pairs)))
    print("Preprocessing labeled validation corpus...")
    attack_val_pairs = loadPairs(voc, attack_val_path, last_only=True)
    print("Loaded {} pairs".format(len(attack_val_pairs)))
    print("Preprocessing labeled analysis corpus...")
    analysis_pairs = loadPairs(voc, analysis_path, last_only=True)
    print("Loaded {} pairs".format(len(analysis_pairs)))
    return attack_train_pairs, attack_val_pairs, analysis_pairs

def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence] + [EOS_token]

def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(False)
            else:
                m[i].append(True)
    return m

# Takes a batch of dialogs (lists of lists of tokens) and converts it into a
# batch of utterances (lists of tokens) sorted by length, while keeping track of
# the information needed to reconstruct the original batch of dialogs
def dialogBatch2UtteranceBatch(dialog_batch):
    utt_tuples = [] # will store tuples of (utterance, original position in batch, original position in dialog)
    for batch_idx in range(len(dialog_batch)):
        dialog = dialog_batch[batch_idx]
        for dialog_idx in range(len(dialog)):
            utterance = dialog[dialog_idx]
            utt_tuples.append((utterance, batch_idx, dialog_idx))
    # sort the utterances in descending order of length, to remain consistent with pytorch padding requirements
    utt_tuples.sort(key=lambda x: len(x[0]), reverse=True)
    # return the utterances, original batch indices, and original dialog indices as separate lists
    utt_batch = [u[0] for u in utt_tuples]
    batch_indices = [u[1] for u in utt_tuples]
    dialog_indices = [u[2] for u in utt_tuples]
    return utt_batch, batch_indices, dialog_indices

# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.BoolTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch, already_sorted=False):
    if not already_sorted:
        pair_batch.sort(key=lambda x: len(x[0]), reverse=True)
    input_batch, output_batch, label_batch, id_batch = [], [], [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
        label_batch.append(pair[2])
        id_batch.append(pair[3])
    dialog_lengths = torch.tensor([len(x) for x in input_batch])
    input_utterances, batch_indices, dialog_indices = dialogBatch2UtteranceBatch(input_batch)
    inp, utt_lengths = inputVar(input_utterances, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    label_batch = torch.FloatTensor(label_batch) if label_batch[0] is not None else None
    return inp, dialog_lengths, utt_lengths, batch_indices, dialog_indices, label_batch, id_batch, output, mask, max_target_len

def batchIterator(voc, source_data, batch_size, shuffle=True):
    cur_idx = 0
    if shuffle:
        random.shuffle(source_data)
    while True:
        if cur_idx >= len(source_data):
            cur_idx = 0
            if shuffle:
                random.shuffle(source_data)
        batch = source_data[cur_idx:(cur_idx+batch_size)]
        # the true batch size may be smaller than the given batch size if there is not enough data left
        true_batch_size = len(batch)
        # ensure that the dialogs in this batch are sorted by length, as expected by the padding module
        batch.sort(key=lambda x: len(x[0]), reverse=True)
        # for analysis purposes, get the source dialogs and labels associated with this batch
        batch_dialogs = [x[0] for x in batch]
        batch_labels = [x[2] for x in batch]
        # convert batch to tensors
        batch_tensors = batch2TrainData(voc, batch, already_sorted=True)
        yield (batch_tensors, batch_dialogs, batch_labels, true_batch_size) 
        cur_idx += batch_size
