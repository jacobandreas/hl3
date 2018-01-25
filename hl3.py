#!/usr/bin/env python3

from nav.task import NavEnv
from craft.task import CraftEnv
import data
import ling
from misc import fakeprof, hlog, util
import training

from collections import Counter
import itertools as it
import logging
import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data as torch_data
from tqdm import tqdm

config = util.Struct()
config.N_SETUP_EXAMPLES = 1000
config.N_EXAMPLES = 20000
config.N_BATCH_EXAMPLES = 10
config.N_BATCH_STEPS = 50
config.N_ROLLOUT_MAX = 1000
config.N_EPOCHS = 20
config.N_LOG = 100
config.CACHE_DIR = '/data/jda/hl3/_cache'
#config.CACHE_DIR = None


ENV = CraftEnv
#ENV = NavEnv

class StateFeaturizer(torch.nn.Module):
    N_OBS = ENV.n_features
    N_HIDDEN = 256

    def __init__(self, dataset):
        super(StateFeaturizer, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(self.N_OBS, self.N_HIDDEN),
            torch.nn.ReLU(),
            torch.nn.Linear(self.N_HIDDEN, self.N_HIDDEN))

    def forward(self, obs):
        return self.layers(obs)

class Policy(torch.nn.Module):
    # TODO auto
    N_WORDVEC = 64
    N_HIDDEN = 256

    def __init__(self, dataset):
        super(Policy, self).__init__()
        self.embed = torch.nn.Linear(len(dataset.vocab), self.N_WORDVEC)
        self.rnn = torch.nn.GRU(
            input_size=self.N_WORDVEC, hidden_size=self.N_HIDDEN, num_layers=1)

        self.predict = torch.nn.Bilinear(self.N_HIDDEN, StateFeaturizer.N_HIDDEN, ENV.n_actions)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, features, batch):
        emb = self.embed(batch.desc)
        _, enc = self.rnn(emb)

        logits = self.predict(enc.squeeze(0), features)
        logprobs = self.log_softmax(logits)
        return logprobs

    def act(self, features, batch):
        probs = self(features, batch).exp().data.cpu().numpy()
        actions = []
        for row in probs:
            actions.append(np.random.choice(ENV.n_actions, p=row))
        return actions

class Splitter(torch.nn.Module):
    def __init__(self, dataset):
        super(Splitter, self).__init__()
        self.predict = torch.nn.Linear(StateFeaturizer.N_HIDDEN, 1)

    def forward(self, features, batch):
        scores = self.predict(features).squeeze(2)
        return scores

class Decoder(torch.nn.Module):
    N_HIDDEN = 256

    def __init__(self, dataset, start_sym, stop_sym):
        super(Decoder, self).__init__()
        self.vocab = dataset.vocab
        self.start_sym = start_sym
        self.stop_sym = stop_sym
        # TODO switch to word embeddings / sparse?
        self.emb = torch.nn.Linear(len(self.vocab), self.N_HIDDEN)
        self.rnn = torch.nn.GRU(
            input_size=self.N_HIDDEN, hidden_size=self.N_HIDDEN, num_layers=1)
        self.out = torch.nn.Linear(self.N_HIDDEN, len(self.vocab))
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    # TODO shape messiness
    def forward(self, state, inp):
        emb = self.emb(inp)
        pre_out, state = self.rnn(emb, state)
        label_logprobs = self.log_softmax(self.out(state[0]))
        return state, label_logprobs

    def decode(self, state):
        n_batch = state.data.shape[1]
        start_id = self.vocab[self.start_sym]
        out = [[start_id] for _ in range(n_batch)]
        inp = [start_id for _ in range(n_batch)]
        done = [False for _ in range(n_batch)]
        for _ in range(20):
            hot_inp = np.zeros((1, n_batch, len(self.vocab)))
            for i, t in enumerate(inp):
                hot_inp[0, i, t] = 1
            hot_inp = Variable(torch.FloatTensor(hot_inp)).cuda() # TODO !!!
            new_state, label_logprobs = self(state, hot_inp)
            new_inp = []
            label_data = label_logprobs.data.cpu().numpy()
            for i, probs in enumerate(label_data):
                #choice = np.random.choice(len(self.vocab), p=np.exp(probs))
                choice = probs.argmax()
                new_inp.append(choice)
                if not done[i]:
                    out[i].append(choice)
                done[i] = done[i] or choice == self.vocab[self.stop_sym]
            state = new_state
            inp = new_inp
        return out

class Model(object):
    def __init__(self, dataset):
        self.featurizer = StateFeaturizer(dataset).cuda()
        self.policy = Policy(dataset).cuda()
        self.splitter = Splitter(dataset).cuda()
        self.describer = Decoder(dataset, ling.START, ling.STOP).cuda()

        self.policy_obj = torch.nn.NLLLoss().cuda()
        self.describer_obj = torch.nn.NLLLoss().cuda()

        self.params = it.chain(
            self.featurizer.parameters(), self.policy.parameters(),
            self.splitter.parameters(), self.describer.parameters())

        self.opt = torch.optim.Adam(self.params, lr=0.001)

        self.dataset = dataset

    def evaluate(self, pol_batch, seg_batch, train=False):
        # policy
        features = self.featurizer(pol_batch.obs)
        #print('good')
        #print(features.shape)
        #print(pol_batch.desc.shape)
        #print(pol_batch.act.shape)
        preds = self.policy(features, pol_batch)
        pol_loss = self.policy_obj(preds, pol_batch.act)

        # splitter
        seg_features = self.featurizer(seg_batch.obs)
        split_scores = self.splitter(seg_features, seg_batch)
        # TODO make a module
        baselines = split_scores.gather(1, seg_batch.final[:, np.newaxis])
        baselined_scores = split_scores - baselines
        augmented_scores = baselined_scores + seg_batch.loss
        margin_losses = torch.nn.functional.relu(augmented_scores)
        bottoms, _ = split_scores.min(1, keepdim=True)
        bottomed_scores = split_scores - bottoms
        split_loss = margin_losses.mean() + .1 * bottomed_scores.mean()
        split_pred_final = split_scores.data.cpu().numpy().argmax(axis=1)
        split_acc = (split_pred_final == seg_batch.final.data.cpu().numpy()).mean()

        # describer
        desc_len = seg_batch.desc.data.shape[0]
        indices = seg_batch.final[:, np.newaxis, np.newaxis]
        desc_features = self.featurizer(seg_batch.last_obs)
        desc_state = desc_features[np.newaxis, ...]
        desc_loss = 0
        for i in range(desc_len - 1):
            desc_state, word_logprobs = self.describer(
                    desc_state, seg_batch.desc[np.newaxis, i, ...])
            desc_loss += self.describer_obj(word_logprobs, seg_batch.desc_target[i])

        if train:
            loss = pol_loss + split_loss + desc_loss
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        pol_score = pol_loss.data.cpu().numpy()[0]
        split_score = split_loss.data.cpu().numpy()[0]
        desc_score = desc_loss.data.cpu().numpy()[0]
        return Counter({
            'pol_nll': pol_score,
            'split_loss': split_score,
            'split_acc': split_acc,
            'desc_nll': desc_score,
        })

    def act(self, pol_batch):
        features = self.featurizer(pol_batch.obs)
        return self.policy.act(features, pol_batch)

    def parse(self, seg_batch):
        assert len(seg_batch.tasks) == 1
        demo = seg_batch.tasks[0].demonstration()
        actions = [a for s, a, s_ in demo]

        # TODO probably move to data
        feature_cache = {}
        def featurize_from(lo):
            if lo in feature_cache:
                return feature_cache[lo]
            state_before = demo[lo][0]
            features = [None] * len(demo)
            for i in range(lo, len(demo)):
                state_after = demo[i][0]
                state = state_after.with_init(state_before)
                features[i] = state.features()
            feature_cache[lo] = features
            return features

        #feature_cache = {}
        #def desc_split(lo, hi, depth):
        #    if lo not in feature_cache:
        #        feature_cache[lo] = featurize_from(lo)
        #    features = feature_cache[lo][lo:hi+1]
        #    split_batch = data.SegmentBatch(
        #        None, None,
        #        Variable(torch.FloatTensor([features])).cuda(),
        #        Variable(torch.FloatTensor([features[-1]])).cuda(),
        #        None, None, None)
        #    last_obs_feats = self.featurizer(split_batch.last_obs)

        #    print(features[-1])

        #    desc = self.describer.decode(last_obs_feats[np.newaxis, ...])[0]
        #    desc = ' '.join(self.dataset.vocab.get(t) for t in desc)
        #    if depth == 0:
        #        return ((lo, hi), desc)

        #    obs_feats = self.featurizer(split_batch.obs)
        #    split_scores = self.splitter(obs_feats, split_batch)
        #    print(split_scores)
        #    split = lo + split_scores.data.cpu().numpy().ravel()[:-1].argmax()
        #    return (
        #        (lo, hi),
        #        desc,
        #        desc_split(lo, split, depth-1),
        #        desc_split(split, hi, depth-1))

        def desc_split(lo, hi, depth):
            assert hi > lo
            features_before = featurize_from(lo)

            candidates = []

            for split in range(lo + 1, hi - 1):
                features_after = featurize_from(split)

                before_batch = data.SegmentBatch(
                    None, None, None,
                    Variable(torch.FloatTensor([features_before[split]])).cuda(),
                    None, None, None)

                after_batch = data.SegmentBatch(
                    None, None, None,
                    Variable(torch.FloatTensor([features_after[hi]])).cuda(),
                    None, None, None)

                obs_feats_before = self.featurizer(before_batch.last_obs)
                obs_feats_after = self.featurizer(after_batch.last_obs)

                desc_before = self.describer.decode(obs_feats_before[np.newaxis, ...])[0]
                desc_after = self.describer.decode(obs_feats_after[np.newaxis, ...])[0]
                hot_desc_before = np.zeros((len(desc_before), split-lo+1, len(self.dataset.vocab)))
                hot_desc_after = np.zeros((len(desc_after), hi-split+1, len(self.dataset.vocab)))
                for i in range(len(desc_before)):
                    hot_desc_before[i, :, desc_before[i]] = 1
                for i in range(len(desc_after)):
                    hot_desc_after[i, :, desc_after[i]] = 1

                before_pol_batch = data.PolicyBatch(
                    Variable(torch.FloatTensor(hot_desc_before)).cuda(),
                    None,
                    Variable(torch.FloatTensor(features_before[lo:split+1])).cuda(),
                    Variable(torch.LongTensor(actions[lo:split] + [self.dataset.env.STOP])).cuda(),
                    None)

                after_pol_batch = data.PolicyBatch(
                    Variable(torch.FloatTensor(hot_desc_after)).cuda(),
                    None,
                    Variable(torch.FloatTensor(features_after[split:hi+1])).cuda(),
                    Variable(torch.LongTensor(actions[split:hi] + [self.dataset.env.STOP])).cuda(),
                    None)

                before_feats = self.featurizer(before_pol_batch.obs)
                after_feats = self.featurizer(after_pol_batch.obs)

                #print('bad')
                #print(before_feats.shape)
                #print(before_pol_batch.desc.shape)
                #print(before_pol_batch.act.shape)
                #print(after_feats.shape)
                #print(after_pol_batch.desc.shape)
                #print(after_pol_batch.act.shape)
                before_preds = self.policy(before_feats, before_pol_batch)
                after_preds = self.policy(after_feats, after_pol_batch)

                before_loss = self.policy_obj(before_preds, before_pol_batch.act)
                after_loss = self.policy_obj(after_preds, after_pol_batch.act)
                candidates.append((
                    before_loss.data.cpu().numpy()[0] +
                        after_loss.data.cpu().numpy()[0],
                    (lo, split, hi),
                    ' '.join(self.dataset.vocab.get(t) for t in desc_before),
                    ' '.join(self.dataset.vocab.get(t) for t in desc_after)
                ))
                #print(before_loss, after_loss)
            score, (lo, split, hi), desc1, desc2 = max(candidates, key=lambda x: x[0])
            return [((lo, split), desc1), ((split, hi), desc2)]

        return desc_split(0, len(demo)-1, 1)

@profile
def main():
    np.random.seed(0)
    dataset, val_dataset, parse_ex = data.get_dataset(ENV, config)
    model = Model(dataset)

    loader = torch_data.DataLoader(
        dataset, batch_size=config.N_BATCH_EXAMPLES, shuffle=True, 
        num_workers=4,
        collate_fn=lambda items: data.collate(items, dataset, config))

    with hlog.task('train'):
        stats = Counter()
        i_iter = 0
        for i_epoch in hlog.loop('epoch_%05d', range(config.N_EPOCHS)):
            for i_batch, (pol_batch, seg_batch) in hlog.loop('batch_%05d', enumerate(loader)):
                if i_iter > 0 and i_iter % config.N_LOG == 0:
                    for k, v in stats.items():
                        hlog.value(k, v / config.N_LOG)
                    with hlog.task('val'):
                        training.validate(model, val_dataset, parse_ex, ENV, config)
                    stats = Counter()

                pol_batch = pol_batch.cuda()
                seg_batch = seg_batch.cuda()
                stats += model.evaluate(pol_batch, seg_batch, train=True)

                i_iter += 1

if __name__ == '__main__':
    main()
