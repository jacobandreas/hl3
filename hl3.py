#!/usr/bin/env python3

from craft.task import CraftEnv
from misc import hlog, util

from collections import namedtuple
import itertools
import logging
import numpy as np
import torch

N_SETUP_EXAMPLES = 100
N_EXAMPLES = 10
N_BATCH_EXAMPLES = 10
N_BATCH_STEPS = 500
N_ITERS = 1000
N_LOG = 5

UNK = 'UNK'

ENV = CraftEnv

vocab = util.Index()

def tokenize(hint, index=False):
    words = hint.lower().split()
    if index:
        toks = [vocab.index(w) for w in words]
    else:
        toks = [vocab[w] or vocab[UNK] for w in words]
    assert None not in toks
    return toks

@hlog.fn("setup")
def setup():
    data = [ENV.sample_task() for _ in range(N_SETUP_EXAMPLES)]
    vocab.index(UNK)
    for datum in data:
        tokenize(datum.desc, index=True)
    with hlog.task('data_size'):
        hlog.log(len(data))
    with hlog.task('vocab_size'):
        hlog.log(len(vocab))

def load_batch(data):
    batch_ids = [np.random.randint(len(data)) for _ in range(N_BATCH_EXAMPLES)]
    batch = [data[i] for i in batch_ids]
    all_labeled_steps = [(task.desc, step) for task in batch for step in task.demonstration()]
    labeled_steps = []
    for _ in range(N_BATCH_STEPS):
        step_id = np.random.randint(len(all_labeled_steps))
        #labeled_steps.append(all_labeled_steps.pop(step_id))
        labeled_steps.append(all_labeled_steps[step_id])

    descs, steps = zip(*labeled_steps)

    # descs
    proc_descs = [tokenize(d) for d in descs]
    max_desc_len = max(len(d) for d in proc_descs)
    desc_data = torch.zeros(max_desc_len, len(descs), len(vocab))
    for i_desc, desc in enumerate(proc_descs):
        for i_tok, tok in enumerate(desc):
            desc_data[i_tok, i_desc, tok] = 1

    act_data = torch.LongTensor([a for s, a, s_ in steps])
    obs_data = torch.FloatTensor([s.features() for s, a, s_ in steps])
    return Batch(
            torch.autograd.Variable(desc_data),
            torch.autograd.Variable(act_data),
            torch.autograd.Variable(obs_data))

Batch = namedtuple('Batch', ['desc', 'act', 'obs'])

class Model(torch.nn.Module):
    N_OBS = 5 * 5 * 5 * 6 * 3
    N_WORDVEC = 64
    N_HIDDEN = 256

    def __init__(self):
        super(Model, self).__init__()
        self.embed = torch.nn.Linear(len(vocab), self.N_WORDVEC)
        self.rnn = torch.nn.GRU(
            input_size=self.N_WORDVEC, hidden_size=self.N_HIDDEN, num_layers=1)

        self.featurize = torch.nn.Sequential(
            torch.nn.Linear(self.N_OBS, self.N_HIDDEN),
            torch.nn.ReLU(),
            torch.nn.Linear(self.N_HIDDEN, self.N_HIDDEN))

        self.predict = torch.nn.Bilinear(self.N_HIDDEN, self.N_HIDDEN, ENV.n_actions)
        self.log_softmax = torch.nn.LogSoftmax()

    def forward(self, batch):
        emb = self.embed(batch.desc)
        _, enc = self.rnn(emb)
        feats = self.featurize(batch.obs)
        logits = self.predict(enc.squeeze(), feats)
        logprobs = self.log_softmax(logits)
        return logprobs

if __name__ == '__main__':
    setup()
    model = Model()
    objective = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    with hlog.task('train'):
        curr_loss = 0
        data = [ENV.sample_task() for _ in range(N_EXAMPLES)]
        for i_iter in range(N_ITERS):
            with hlog.task('iter_%04d' % i_iter):
                if i_iter > 0 and i_iter % N_LOG == 0:
                    hlog.value('loss', curr_loss / N_LOG)
                    curr_loss = 0
                    data = [ENV.sample_task() for _ in range(N_EXAMPLES)]

                optimizer.zero_grad()
                batch = load_batch(data)
                logprobs = model(batch)
                loss = objective(logprobs, batch.act)
                loss.backward()
                optimizer.step()
                curr_loss += loss.data.numpy()[0]
