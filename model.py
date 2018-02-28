import data
import ling

from collections import Counter
import itertools as it
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable

class StateFeaturizer(nn.Module):
    N_HIDDEN = 256

    def __init__(self, env, dataset):
        super().__init__()
        n_obs = env.n_features
        self._predict = nn.Sequential(
            nn.Linear(n_obs, self.N_HIDDEN),
            nn.ReLU(),
            nn.Linear(self.N_HIDDEN, self.N_HIDDEN))

    def forward(self, obs):
        return self._predict(obs)

class Policy(nn.Module):
    # TODO magic
    N_WORDVEC = 64
    N_HIDDEN = 256

    def __init__(self, env, dataset):
        super().__init__()
        self._n_actions = env.n_actions
        self._embed = nn.Linear(len(dataset.vocab), self.N_WORDVEC)
        self._encoder = nn.GRU(
            input_size=self.N_WORDVEC, hidden_size=self.N_HIDDEN, num_layers=1)
        self._decoder = Decoder(dataset.vocab, ling.START, ling.STOP)
        self._predict = nn.Bilinear(self.N_HIDDEN, StateFeaturizer.N_HIDDEN,
            self._n_actions)
        self._softmax = nn.Softmax(dim=1)

    def forward(self, features, batch):
        emb = self._embed(batch.desc_in)
        _, enc = self._encoder(emb)
        enc = enc.squeeze(0)
        act_logits = self._predict(enc, features)
        if batch.desc_out is None:
            return act_logits, None
        else:
            desc_logits = self._decoder(enc, batch.desc_out)
            return act_logits, desc_logits

    def act(self, features, batch):
        probs = self._softmax(self(features, batch))
        probs = probs.data.cpu().numpy()
        actions = []
        for row in probs:
            actions.append(np.random.choice(self._n_actions, p=row))
        return actions

class Decoder(nn.Module):
    N_HIDDEN = 256

    def __init__(self, vocab, start_sym, stop_sym):
        super().__init__()
        self._vocab = vocab
        self._start_id = vocab[start_sym]
        self._stop_id = vocab[stop_sym]

        self._embed = nn.Linear(len(vocab), self.N_HIDDEN)
        self._rnn = nn.GRU(
            input_size=self.N_HIDDEN, hidden_size=self.N_HIDDEN, num_layers=1)
        self._predict = nn.Linear(self.N_HIDDEN, len(vocab))
        self._softmax = nn.Softmax(dim=1)

    def forward(self, state, inp):
        emb = self._embed(inp)
        rep, enc = self._rnn(emb, state)
        logits = self._predict(rep)
        return enc, logits

    def decode(self, init_state, max_len):
        n_stack, n_batch, _ = init_state.shape
        out = [[self._start_id] for _ in range(n_batch)]
        tok_inp = [self._start_id for _ in range(n_batch)]
        done = [False for _ in range(n_batch)]
        state = init_state
        for _ in range(max_len):
            hot_inp = np.zeros((1, n_batch, len(self._vocab)))
            for i, t in enumerate(tok_inp):
                hot_inp[0, i, t] = 1
            hot_inp = Variable(torch.FloatTensor(hot_inp))
            if init_state.is_cuda:
                hot_inp = hot_inp.cuda()
            new_state, label_logits = self(state, hot_inp)
            label_logits = label_logits.squeeze(1)
            label_probs = self._softmax(label_logits)
            label_probs = label_probs.data.cpu().numpy()
            new_tok_inp = []
            for i, row in enumerate(label_probs):
                tok = row.argmax()
                new_tok_inp.append(tok)
                if not done[i]:
                    out[i].append(tok)
                done[i] = done[i] or tok == self._stop_id
            state = new_state
            tok_inp = new_tok_inp
        return out

class TopDownParser(object):
    def __init__(self, model, env, dataset):
        self._featurizer = model._featurizer
        self._policy = model._policy
        self._policy_prob = model._policy_prob
        self._describer = model._describer

        self._env = env
        self._dataset = dataset

    def parse(self, seq_batch):
        demos = [task.demonstration() for task in seq_batch.tasks]

        feature_cache = {}
        for d in range(1): #range(len(demos)):
            demo = demos[d]
            for i in range(len(demo)):
                state_before, _, _ = demo[i]
                for j in range(i, len(demo)):
                    state, _, _ = demo[j]
                    state = state.with_init(state_before)
                    feature_cache[d, i, j] = state.features()

        def score_span(d, i, j, desc):
            # TODO util function
            demo = demos[d]
            hot_desc = np.zeros((len(desc), j-i+1, len(self._dataset.vocab)))
            for p, t in enumerate(desc):
                hot_desc[p, :, t] = 1
            features = [feature_cache[d, i, k] for k in range(i, j+1)]
            actions = [demo[k][1] for k in range(i, j)] + [self._env.STOP]

            # TODO Batch.of_x
            step_batch = data.StepBatch(
                Variable(torch.FloatTensor(features)),
                Variable(torch.LongTensor(actions)),
                Variable(torch.FloatTensor(hot_desc)),
                None, None, None)
            rep = self._featurizer(step_batch.obs)
            act_logits, _ = self._policy(rep, step_batch)
            score = self._policy_prob(act_logits, step_batch.act)
            return score

        def propose_desc(d, i, j, n):
            feats = [feature_cache[d, i, j] for _ in range(n)]
            feats_var = Variable(torch.FloatTensor(feats))
            reps = self._featurizer(feats_var)
            # TODO magic
            descs = self._describer.decode(reps.unsqueeze(0), 20)
            return descs

        def render(desc):
            return ' '.join(self._dataset.vocab.get(t) for t in desc)

        for d, demo in enumerate(demos):
            if len(demo) < 3:
                break
            descs = []
            scores = []
            splits = list(range(1, len(demo)-1))
            for k in splits:
                desc1, = propose_desc(d, 0, k, 1)
                desc2, = propose_desc(d, k, len(demo)-1, 1)
                descs.append((desc1, desc2))
                score = (
                    score_span(d, 0, k, desc1)
                    + score_span(d, k, len(demo)-1, desc2))
                scores.append(score.data.cpu().numpy()[0])
            # TODO argmin on gpu?
            i_split = np.asarray(scores).argmin()
            split = splits[i_split]
            d1, d2 = descs[i_split]
            print(seq_batch.tasks[d].desc)
            print(render(d1), render(d2))
            break

class Model(object):
    def __init__(self, env, dataset, config):
        self._featurizer = StateFeaturizer(env, dataset)
        self._policy = Policy(env, dataset)
        self._flat_policy = Policy(env, dataset)
        self._describer = Decoder(dataset.vocab, ling.START, ling.STOP)

        self._policy_obj = nn.CrossEntropyLoss()
        self._policy_prob = nn.CrossEntropyLoss(size_average=False)
        self._describer_obj = nn.CrossEntropyLoss()

        self._parser = TopDownParser(self, env, dataset)

        if config.GPU:
            for module in [
                    self._featurizer, self._policy, self._flat_policy,
                    self._describer, self._parser, self._policy_obj,
                    self._policy_prob, self._describer_obj]:
                module.cuda()

        params = it.chain(
            self._featurizer.parameters(), self._policy.parameters(),
            self._describer.parameters())
        # TODO magic
        self._opt = optim.Adam(params, lr=0.001)

    def parse(self, seq_batch):
        return self._parser.parse(seq_batch)

    def train_policy(self, step_batch):
        rep = self._featurizer(step_batch.obs)
        act_logits, _ = self._policy(rep, step_batch)
        loss = self._policy_obj(act_logits, step_batch.act)
        self._opt.zero_grad()
        loss.backward()
        self._opt.step()
        return Counter({'pol_loss': loss.data.cpu().numpy()[0]})

    def train_helpers(self, seq_batch):
        rep = self._featurizer(seq_batch.seq_obs)
        _, desc_logits = self._describer(rep.unsqueeze(0), seq_batch.desc)
        n_tok, n_batch, n_pred = desc_logits.shape
        desc_loss = self._describer_obj(
            desc_logits.view(n_tok * n_batch, n_pred),
            seq_batch.desc_target.view(n_tok * n_batch))
        loss = desc_loss
        self._opt.zero_grad()
        loss.backward()
        self._opt.step()
        return Counter({'desc_loss': desc_loss.data.cpu().numpy()[0]})