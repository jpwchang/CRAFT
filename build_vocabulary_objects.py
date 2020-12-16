import unicodedata
import json
import os
import nltk.tokenize
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from model.config import *

VOCAB_SIZE = 50000
NUM_CHUNKS = 200

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def tokenize(text):
    tokenizer = nltk.tokenize.RegexpTokenizer(pattern=r'\w+|[^\w\s]')
    # simplify the problem space by considering only ASCII data
    cleaned_text = unicodeToAscii(text.lower())

    # if the resulting string is empty, nothing else to do
    if not cleaned_text.strip():
        return []
    
    return tokenizer.tokenize(cleaned_text)

def count_tokens_in_chunk(idx, chunk):
    print("Processing chunk", idx)
    counts = Counter()

    for dialog in chunk:
        for utterance in dialog:
            tokens = tokenize(utterance["text"])
            counts += Counter(tokens)

    return counts

def main():
    print("Loading training dataset...")
    with open(os.path.join(repo_dir, "nn_input_data", corpus_name, "train_processed_dialogs.txt")) as fp:
        dialogs = [json.loads(line) for line in fp]
    
    chunk_size = len(dialogs) // NUM_CHUNKS
    dialog_chunks = [dialogs[i:i+chunk_size] for i in range(0, len(dialogs), chunk_size)]

    global_counts = Counter()

    with Pool(40) as p:
        counts_per_dialog = p.starmap(count_tokens_in_chunk, list(enumerate(dialog_chunks)))
    print("Merging chunks...")
    for dialog_counts in counts_per_dialog:
        global_counts += dialog_counts

    print("Truncating to vocabulary size", VOCAB_SIZE)
    kept_counts = global_counts.most_common(VOCAB_SIZE)

    print("Converting to dicts")
    word2index = {"UNK": UNK_token}
    index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: "UNK"}
    num_words = 4  # Count SOS, EOS, PAD, UNK
    for token, _ in kept_counts:
        word2index[token] = num_words
        index2word[num_words] = token
        num_words += 1

    print("Dumping")
    with open(os.path.join(repo_dir, "nn_preprocessing", corpus_name, "word2index.json"), "w") as fp:
        json.dump(word2index, fp)
    with open(os.path.join(repo_dir, "nn_preprocessing", corpus_name, "index2word.json"), "w") as fp:
        json.dump(index2word, fp)

if __name__ == "__main__":
    main()