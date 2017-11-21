#!/usr/bin/env python3

from craft.task import CraftEnv
from craft.builder import BlockType
from misc import hlog, util, fakeprof

from collections import namedtuple
import itertools
import logging
import numpy as np
import os
import pickle
import torch
import torch.utils.data
from tqdm import tqdm

N_SETUP_EXAMPLES = 1000
N_EXAMPLES = 10000
N_BATCH_EXAMPLES = 10
N_BATCH_STEPS = 500
N_ROLLOUT_MAX = 1000
N_EPOCHS = 100
N_LOG = 50
CACHE_DIR = '/data/jda/hl3/_cache'

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

@hlog.fn('setup')
def setup():
    data = [ENV.sample_task() for _ in range(N_SETUP_EXAMPLES)]
    vocab.index(UNK)
    for datum in data:
        print(datum.desc)
        tokenize(datum.desc, index=True)
    with hlog.task('data_size'):
        hlog.log(len(data))
    with hlog.task('vocab_size'):
        hlog.log(len(vocab))

Batch = namedtuple('Batch', ['desc', 'act', 'obs'])

@profile
def load_batch(tasks, actions, features):
    desc = []
    act = []
    obs = []
    for _ in range(N_BATCH_STEPS):
        task_id = np.random.randint(len(tasks))
        step_id = np.random.randint(len(actions[task_id]))
        desc.append(tasks[task_id].desc)
        act.append(actions[task_id][step_id])
        obs.append(features[task_id][step_id])

    # descs
    proc_desc = [tokenize(d) for d in desc]
    max_desc_len = max(len(d) for d in proc_desc)
    desc_data = torch.zeros(max_desc_len, len(desc), len(vocab))
    for i_desc, desc in enumerate(proc_desc):
        for i_tok, tok in enumerate(desc):
            desc_data[i_tok, i_desc, tok] = 1

    act_data = torch.LongTensor(act)
    obs_data = torch.FloatTensor(obs)
    return Batch(
            torch.autograd.Variable(desc_data),
            torch.autograd.Variable(act_data),
            torch.autograd.Variable(obs_data))

@profile
def rollout(model, task):
    desc = tokenize(task.desc)
    desc_data = torch.zeros(len(desc), 1, len(vocab))
    for i, tok in enumerate(desc):
        desc_data[i, 0, tok] = 1
    desc_var = torch.autograd.Variable(desc_data.cuda())

    state = task.init_state
    steps = []
    for _ in range(N_ROLLOUT_MAX):
        obs_data = [state.features()]
        obs_data = torch.FloatTensor(obs_data)
        batch = Batch(
                desc_var,
                None,
                torch.autograd.Variable(obs_data.cuda()))
        action, = model.act(batch)
        s_ = state.step(action)
        steps.append((state, action, s_))
        state = s_
        if action == ENV.STOP:
            break
    return steps

class Model(torch.nn.Module):
    # TODO auto
    N_OBS = 5 * 5 * 5 * 7 * 3 * 2 + 3 + len(BlockType.enumerate())
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

        logits = self.predict(enc.squeeze(0), feats)
        logprobs = self.log_softmax(logits)
        return logprobs

    def act(self, batch):
        probs = self.forward(batch).exp().data.cpu().numpy()
        actions = []
        for row in probs:
            actions.append(np.random.choice(ENV.n_actions, p=row))
            #actions.append(row.argmax())
        return actions

class Dataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir):
        with open(os.path.join(CACHE_DIR, 'tasks.pkl'), 'rb') as task_f:
            self.tasks = pickle.load(task_f)
        with open(os.path.join(CACHE_DIR, 'actions.pkl'), 'rb') as action_f:
            self.actions = pickle.load(action_f)
        assert len(self.tasks) == len(self.actions)

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, i):
        features = np.load(os.path.join(CACHE_DIR, 'features', 'path%d.npy' % i))
        return self.tasks[i], self.actions[i], features

def get_dataset():
    if os.path.exists(CACHE_DIR):
        return Dataset(CACHE_DIR)

    os.mkdir(CACHE_DIR)
    tasks = [ENV.sample_task() for _ in range(N_EXAMPLES)]
    demonstrations = [task.demonstration() for task in tasks]
    actions = [[a for s, a, s_ in demo] for demo in demonstrations]
    with open(os.path.join(CACHE_DIR, 'tasks.pkl'), 'wb') as task_f:
        pickle.dump(tasks, task_f)
    with open(os.path.join(CACHE_DIR, 'actions.pkl'), 'wb') as action_f:
        pickle.dump(actions, action_f)

    os.mkdir(os.path.join(CACHE_DIR, 'features'))
    for i_task, task in tqdm(list(enumerate(tasks))):
        demo = demonstrations[i_task]
        features = []
        for s, a, s_ in demo:
            features.append(s.features())
        np.save(
            os.path.join(CACHE_DIR, 'features', 'path%d.npy' % i_task),
            np.asarray(features))

    return Dataset(CACHE_DIR)

def validate(model):
    eval_task = ENV.sample_task()
    steps = rollout(model, eval_task)
    actions = [s[1] for s in steps]
    last_state = steps[-1][0]
    last_scene = last_state.to_scene()
    hlog.value('desc', eval_task.desc)
    hlog.value('sampled', ' '.join(str(a) for a in actions))
    hlog.value('gold', ' '.join(str(a) for s, a, s_ in eval_task.demonstration()))
    with open('vis/before.json', 'w') as scene_f:
        eval_task.scene_before.dump(scene_f)
    with open('vis/after.json', 'w') as scene_f:
        last_scene.dump(scene_f)
    with open('vis/after_gold.json', 'w') as scene_f:
        eval_task.scene_after.dump(scene_f)

@profile
def main():
    setup()
    model = Model().cuda()
    objective = torch.nn.NLLLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    np.random.seed(0)
    dataset = get_dataset()

    def collate(items):
        tasks, actions, features = zip(*items)
        return load_batch(tasks, actions, features)

    loader = torch.utils.data.DataLoader(
            dataset, batch_size=N_BATCH_EXAMPLES, shuffle=True, num_workers=2,
            collate_fn=collate)

    with hlog.task('train'):
        curr_loss = 0
        i_iter = 0
        for i_epoch in hlog.loop('epoch_%05d', range(N_EPOCHS)):
            for i_batch, batch in hlog.loop('batch_%05d', enumerate(loader)):
                if i_iter > 0 and i_iter % N_LOG == 0:
                    hlog.value('loss', curr_loss / N_LOG)
                    curr_loss = 0
                    validate(model)

                batch = Batch(
                        batch.desc.cuda(), batch.act.cuda(), batch.obs.cuda())
                logprobs = model(batch)
                loss = objective(logprobs, batch.act)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                curr_loss += loss.data.cpu().numpy()[0]
                i_iter += 1

if __name__ == '__main__':
    main()
