START = "<s>"
STOP = "</s>"
UNK = 'UNK'

def tokenize(hint, vocab, index=False):
    words = [START] + hint.lower().split() + [STOP]
    if index:
        toks = [vocab.index(w) for w in words]
    else:
        toks = [vocab[w] or vocab[UNK] for w in words]
    assert None not in toks
    return toks
