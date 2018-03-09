import data
import ling

from collections import Counter
import gflags
import itertools as it
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable

FLAGS = gflags.FLAGS

class StateFeaturizer(nn.Module):
    N_HIDDEN = 256

    def __init__(self, env, dataset):
        super().__init__()
        n_obs = env.n_features
        self._predict = nn.Sequential(
            nn.Linear(2 * n_obs, self.N_HIDDEN),
            nn.ReLU(),
            nn.Linear(self.N_HIDDEN, self.N_HIDDEN))

    def forward(self, init_obs, obs):
        return self._predict(torch.cat((init_obs, obs), 1))

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
            desc_logits = self._decoder(enc.unsqueeze(0), batch.desc_out)
            return act_logits, desc_logits

    def act(self, features, batch):
        probs = self._softmax(self(features, batch))
        probs = probs.data.cpu().numpy()
        actions = []
        for row in probs:
            actions.append(np.random.choice(self._n_actions, p=row))
        return actions

class Segmenter(nn.Module):
    def __init__(self, env, dataset):
        super().__init__()
        self._predict = nn.Linear(StateFeaturizer.N_HIDDEN, 1)

    def forward(self, features):
        return self._predict(features).squeeze(-1)

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

    @profile
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
            if all(done):
                break
        return out

class TopDownParser(object):
    def __init__(self, model, env, dataset):
        self._featurizer = model._featurizer
        self._policy = model._policy
        self._policy_prob = model._policy_prob
        self._describer = model._describer
        self._segmenter = model._segmenter

        self._env = env
        self._dataset = dataset

    @profile
    def parse(self, seq_batch):
        demos = [task.demonstration() for task in seq_batch.tasks]

        feature_cache = {}
        for d, demo in enumerate(demos):
            init_obs = seq_batch.init_obs[d, :].unsqueeze(0).expand((len(demo), -1))
            mid_obs = torch.stack([seq_batch.all_obs[d, i, :] for i in range(len(demo))])
            end_obs = seq_batch.last_obs[d, :].unsqueeze(0).expand((len(demo), -1))
            reps1 = self._featurizer(init_obs, mid_obs)
            reps2 = self._featurizer(mid_obs, end_obs)
            for i in range(1, len(demo)-1):
                feature_cache[d, 0, i] = reps1[i, :]
                feature_cache[d, i, len(demo)-1] = reps2[i, :]
        def get_feats(d, i, j):
            assert (d, i, j) in feature_cache, (d, i, j)
            return feature_cache[d, i, j]

        def score_span(d, pi, pdesc, i, j, desc):
            # TODO util function
            demo = demos[d]
            hot_desc = np.zeros((max(len(desc), len(pdesc)), j-i+1+1, len(self._dataset.vocab)))
            for p, t in enumerate(pdesc):
                hot_desc[p, 0, t] = 1
            for p, t in enumerate(desc):
                hot_desc[p, 1:, t] = 1

            top_feats = [seq_batch.all_obs[d, i, :]]
            top_init_feats = [seq_batch.all_obs[d, pi, :]]
            top_actions = [self._env.SAY]
            top_desc_out_mask = [1]
            top_desc = [desc]

            bot_feats = [seq_batch.all_obs[d, k, :] for k in range(i, j+1)]
            bot_init_feats = [seq_batch.all_obs[d, i, :] for _ in range(i, j+1)]
            bot_actions = [demo[k][1] for k in range(i, j)] + [self._env.STOP]
            bot_desc_out_mask = [0 for _ in bot_actions]
            bot_desc = [[] for _ in bot_actions]

            feats = torch.stack(top_feats + bot_feats)
            init_feats = torch.stack(top_init_feats + bot_init_feats)
            actions = top_actions + bot_actions
            desc_out_mask = top_desc_out_mask + bot_desc_out_mask
            desc_out, desc_out_target = data.load_desc_data(
                top_desc + bot_desc, self._dataset, target=True, tokenize=False)

            # TODO Batch.of_x
            step_batch = data.StepBatch(
                init_feats,
                feats,
                Variable(torch.LongTensor(actions)),
                None,
                Variable(torch.FloatTensor(hot_desc)),
                Variable(torch.FloatTensor(desc_out_mask)),
                Variable(torch.FloatTensor(desc_out)),
                Variable(torch.LongTensor(desc_out_target)))
            if next(self._policy.parameters()).is_cuda:
                step_batch = step_batch.cuda()

            rep = self._featurizer(step_batch.init_obs, step_batch.obs)
            act_logits, _ = self._policy(rep, step_batch)
            score = self._policy_prob(act_logits, step_batch.act)
            return score

        def propose_desc(d, i, j, n):
            reps = get_feats(d, i, j).unsqueeze(0).expand((n, -1))
            # TODO magic
            descs = self._describer.decode(reps.unsqueeze(0), 20)
            return descs

        def propose_splits(d, n):
            demo = demos[d]
            # TODO expand
            first_reps = torch.stack(
                [get_feats(d, 0, i) for i in range(1, len(demo)-1)])
            second_reps = torch.stack(
                [get_feats(d, i, len(demo)-1) for i in range(1, len(demo)-1)])
            splits = self._segmenter(first_reps)  + self._segmenter(second_reps)
            _, top = splits.topk(min(n, splits.shape[0]), sorted=False)
            return 1 + top

        def render(desc):
            return ' '.join(self._dataset.vocab.get(t) for t in desc)

        # 0   1   2   3   4   5   6   7   8   9
        # s a s a s a s a s a s a s a s a s a s STOP
        #
        # (0, 5)
        # final obs is s5
        # replace a5 with STOP
        # 
        # (5, 9)
        # final obs is s9
        # replace a9 with STOP

        out_descs = []
        out_actions = []
        out_states = []

        for d, demo in enumerate(demos):
            if len(demo) < 3:
                break
            descs = []
            scores = []
            # TODO magic
            splits = propose_splits(d, 5).data.cpu().numpy()
            # TODO gross
            pdesc = ling.tokenize(seq_batch.tasks[d].desc, self._dataset.vocab)
            for k in splits:
                # TODO batch
                desc1, = propose_desc(d, 0, k, 1)
                desc2, = propose_desc(d, k, len(demo)-1, 1)
                descs.append((desc1, desc2))
                score = (
                    score_span(d, 0, pdesc, 0, k, desc1)
                    + score_span(d, 0, pdesc, k, len(demo)-1, desc2))
                scores.append(score.data.cpu().numpy()[0])
            # TODO argmin on gpu?
            i_split = np.asarray(scores).argmin()
            split = splits[i_split]
            d1, d2 = descs[i_split]
            actions = [t[1] for t in demo]

            # TODO HORRIBLE
            out_descs.append(render(d1))
            out_actions.append(actions[0:split] + [self._env.STOP])
            out_states.append(seq_batch.all_obs[d, 0:split+1, :].data)

            out_descs.append(render(d2))
            assert actions[len(demo)-1] == self._env.STOP
            out_actions.append(actions[split:len(demo)-1] + [self._env.STOP])
            out_states.append(seq_batch.all_obs[d, split:len(demo), :].data)

        return data.ParseBatch(out_descs, out_actions, out_states)

            #print(seq_batch.tasks[d].desc)
            #print(splits)
            #print(actions[:split], actions[split:])
            #print(render(d1), render(d2))

class Model(object):
    def __init__(self, env, dataset):
        self._featurizer = StateFeaturizer(env, dataset)
        self._policy = Policy(env, dataset)
        self._flat_policy = Policy(env, dataset)
        self._describer = Decoder(dataset.vocab, ling.START, ling.STOP)
        self._segmenter = Segmenter(env, dataset)

        self._policy_obj = nn.CrossEntropyLoss()
        self._policy_prob = nn.CrossEntropyLoss(size_average=False)
        self._describer_obj = nn.CrossEntropyLoss()
        self._segmenter_obj = nn.BCEWithLogitsLoss()

        if FLAGS.gpu:
            for module in [
                    self._featurizer, self._policy, self._flat_policy,
                    self._describer, self._segmenter, self._policy_obj,
                    self._policy_prob, self._describer_obj, self._segmenter_obj]:
                module.cuda()

        self._parser = TopDownParser(self, env, dataset)

        params = it.chain(
            self._featurizer.parameters(), self._policy.parameters(),
            self._describer.parameters(), self._segmenter.parameters())
        # TODO magic
        self._opt = optim.Adam(params, lr=0.001)

    def parse(self, seq_batch):
        return self._parser.parse(seq_batch)

    def train_step(self, step_batch):
        rep = self._featurizer(step_batch.init_obs, step_batch.obs)
        act_logits, _ = self._policy(rep, step_batch)
        pol_loss = self._policy_obj(act_logits, step_batch.act)
        seg_loss = self._segmenter_obj(self._segmenter(rep), step_batch.final)
        loss = pol_loss + seg_loss
        self._opt.zero_grad()
        loss.backward()
        self._opt.step()
        return Counter({'pol_loss': loss.data.cpu().numpy()[0]})

    def train_seq(self, seq_batch):
        rep = self._featurizer(seq_batch.init_obs, seq_batch.last_obs)
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
