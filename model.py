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

def unwrap(var):
    return var.data.cpu().numpy()

#class StateFeaturizer(nn.Module):
#    N_HIDDEN = 256
#
#    def __init__(self, env, dataset):
#        super().__init__()
#        n_obs = env.n_features
#        self._predict = nn.Sequential(
#            nn.Linear(2 * n_obs, self.N_HIDDEN),
#            nn.ReLU(),
#            nn.Linear(self.N_HIDDEN, self.N_HIDDEN),
#            nn.ReLU(),
#            nn.Linear(self.N_HIDDEN, self.N_HIDDEN),
#            )
#
#    def forward(self, init_obs, obs):
#        return self._predict(torch.cat((obs, obs - init_obs), 1))

class WorldFeaturizer(nn.Module):
    N_HIDDEN_1 = 128
    N_HIDDEN_2 = 256
    N_HIDDEN = N_HIDDEN_2
    KERNEL_1 = 5
    KERNEL_2 = 1
    PAD_1 = 2
    PAD_2 = 0

    def __init__(self, env, dataset):
        super().__init__()
        self._conv = nn.Sequential(
            nn.Conv3d(
                2*env.n_world_obs, self.N_HIDDEN_1, self.KERNEL_1,
                padding=self.PAD_1),
            nn.ReLU(),
            nn.Conv3d(
                self.N_HIDDEN_1, self.N_HIDDEN_2, self.KERNEL_2,
                padding=self.PAD_2))

    def forward(self, init_obs, obs):
        in_obs = torch.cat((obs, obs-init_obs), 1)
        conv_feats = self._conv(in_obs)
        n_batch = obs.shape[0]
        pool_feats, _ = conv_feats.view(n_batch, self.N_HIDDEN_2, -1).max(2)
        return pool_feats, conv_feats

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
        #self._decoder = Decoder(dataset.vocab, ling.START, ling.STOP)
        # TODO overkill?
        self._predict_act = nn.Bilinear(self.N_HIDDEN, WorldFeaturizer.N_HIDDEN,
            self._n_actions)
        #self._predict = nn.Bilinear(self.N_HIDDEN, StateFeaturizer.N_HIDDEN,
        #    self._n_actions)
        self._act_softmax = nn.Softmax(dim=1)
        self._act_pos_softmax = nn.Softmax(dim=1)

    def forward(self, feats, conv_feats, batch):
        emb = self._embed(batch.desc_in)
        _, enc = self._encoder(emb)
        enc = enc.squeeze(0)
        act_logits = self._predict_act(enc, feats)

        n_batch, n_feats, _, _, _ = conv_feats.shape
        tile_enc = enc.view(n_batch, n_feats, 1, 1, 1).expand_as(conv_feats)
        act_pos_logits = (conv_feats * tile_enc).sum(1).view(n_batch, -1)
        if batch.desc_out is None:
            return act_logits, act_pos_logits, None
        else:
            desc_logits = self._decoder(enc.unsqueeze(0), batch.desc_out)
            return act_logits, act_pos_logits, desc_logits

class Segmenter(nn.Module):
    def __init__(self, env, dataset):
        super().__init__()
        self._predict = nn.Linear(WorldFeaturizer.N_HIDDEN, 1)

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
            label_logits = label_logits.squeeze(0)
            label_probs = unwrap(self._softmax(label_logits))
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
        self._policy_prob = model._policy_act_prob
        self._desc_prob = model._desc_prob
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
            reps1, _ = self._featurizer(init_obs, mid_obs)
            reps2, _ = self._featurizer(mid_obs, end_obs)
            for i in range(1, len(demo)-1):
                feature_cache[d, 0, i] = reps1[i, :]
                feature_cache[d, i, len(demo)-1] = reps2[i, :]
        def get_feats(d, i, j):
            assert (d, i, j) in feature_cache, (d, i, j)
            return feature_cache[d, i, j]

        # TODO bad score_parent
        def score_span(d, pi, pdesc, i, j, desc, score_parent=True):
            # TODO util function
            demo = demos[d]
            #hot_desc = np.zeros((max(len(desc), len(pdesc)), j-i+1+1, len(self._dataset.vocab)))
            hot_desc = np.zeros((max(len(desc), len(pdesc)), 1+1, len(self._dataset.vocab)))
            for p, t in enumerate(pdesc):
                hot_desc[p, 0, t] = 1
            for p, t in enumerate(desc):
                hot_desc[p, 1:, t] = 1

            top_obs = [seq_batch.all_obs[d, i, :]]
            top_init_obs = [seq_batch.all_obs[d, pi, :]]
            top_actions = [self._env.SAY]
            top_desc_out_mask = [1]
            top_desc = [desc]

            #feats = [seq_batch.all_obs[d, k, :] for k in range(i, j+1)]
            #init_feats = [seq_batch.all_obs[d, i, :] for _ in range(i, j+1)]
            #actions = [demo[k][1] for k in range(i, j)] + [self._env.STOP]
            #desc_out_mask = [0 for _ in actions]
            #desc = [[None] for _ in actions]
            obs = [seq_batch.all_obs[d, j, :]]
            init_obs = [seq_batch.all_obs[d, i, :]]
            actions = [self._env.STOP]
            desc_out_mask = [0]
            desc = [[None]]

            if score_parent:
                obs = top_obs + obs
                init_obs = top_init_obs + init_obs
                actions = top_actions + actions
                desc_out_mask = top_desc_out_mask + desc_out_mask
                desc = top_desc + desc
            else:
                hot_desc = hot_desc[:, 1:, ...]

            obs = torch.stack(obs)
            init_obs = torch.stack(init_obs)
            desc_out, desc_out_target = data.load_desc_data(
                desc, self._dataset, target=True, tokenize=False)

            # TODO Batch.of_x
            step_batch = data.StepBatch(
                init_obs,
                obs,
                Variable(torch.LongTensor(actions)),
                None,
                Variable(torch.FloatTensor(hot_desc)),
                Variable(torch.FloatTensor(desc_out_mask)),
                Variable(torch.FloatTensor(desc_out)),
                Variable(torch.LongTensor(desc_out_target)))
            if next(self._policy.parameters()).is_cuda:
                step_batch = step_batch.cuda()

            feats, conv_feats = self._featurizer(step_batch.init_obs, step_batch.obs)
            act_logits, (_, desc_logits) = self._policy(feats, conv_feats, step_batch)
            scores = self._policy_prob(act_logits, step_batch.act)
            return scores

        #def propose_desc(d, i, j, n):
        #    reps = get_feats(d, i, j).unsqueeze(0).expand((n, -1))
        #    # TODO magic
        #    descs = self._describer.decode(reps.unsqueeze(0), 20)
        #    return descs
        def propose_descs(indices, n):
            reps = torch.cat(
                [get_feats(*index).unsqueeze(0).expand((n, -1)) 
                    for index in indices],
                0)
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
            return ' '.join(self._dataset.vocab.get(t) for t in desc[1:-1])

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
            # TODO magic
            splits = unwrap(propose_splits(d, 5))
            # TODO gross
            pdesc = ling.tokenize(seq_batch.tasks[d].desc, self._dataset.vocab)

            parent_scores = score_span(d, 0, pdesc, 0, len(demo)-1, pdesc, score_parent=False)

            indices = [[(d, 0, k), (d, k, len(demo)-1)] for k in splits]
            indices = sum(indices, [])
            descs = propose_descs(indices, 1)
            desc_pairs = [(descs[2*i], descs[2*i+1]) for i in range(len(splits))]

            pick_scores = []
            pick_splits = []
            pick_descs = []
            assert len(splits) == len(desc_pairs)
            for k, (desc1, desc2) in zip(splits, desc_pairs):
                s1c, = unwrap(score_span(d, 0, pdesc, 0, k, desc1).sum())
                s2c, = unwrap(score_span(d, 0, pdesc, k, len(demo)-1, desc2).sum())

                s1p, = unwrap(parent_scores[0:k].sum())
                s2p, = unwrap(parent_scores[0:k].sum())

                pick_desc = [None, None]

                if s1c > s1p:
                    s1 = s1c
                    pick_desc[0] = desc1
                else:
                    s1 = s1p

                if s2c > s1p:
                    s2 = s2c
                    pick_desc[1] = desc2
                else:
                    s2 = s2p

                pick_scores.append(s1 + s2)
                pick_splits.append(k)
                pick_descs.append(tuple(pick_desc))
                #score_parts.append((s1, s2, s1a, s2a, s2b))

            # TODO argmin on gpu?
            i_split = np.asarray(pick_scores).argmin()
            split = pick_splits[i_split]
            score = pick_scores[i_split]
            d1, d2 = pick_descs[i_split]
            actions = [t[1] for t in demo]

            if not (d1 is None and d2 is None) and np.random.random() < 0.02:
                print(
                    render(pdesc), 
                    ':', 
                    render(d1) if d1 else '_', 
                    '>', 
                    render(d2) if d2 else '_')
                print(actions[:split], actions[split:-1])
                print()

            # TODO HORRIBLE
            if d1 is not None:
                out_descs.append(render(d1))
                out_actions.append(actions[0:split] + [self._env.STOP])
                out_states.append(seq_batch.all_obs[d, 0:split+1, :].data)

            if d2 is not None:
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
        self._env = env

        self._featurizer = WorldFeaturizer(env, dataset)
        self._policy = Policy(env, dataset)
        #self._flat_policy = Policy(env, dataset)
        self._describer = Decoder(dataset.vocab, ling.START, ling.STOP)
        self._segmenter = Segmenter(env, dataset)

        self._policy_act_obj = nn.CrossEntropyLoss()
        self._policy_act_prob = nn.CrossEntropyLoss(reduce=False)
        self._policy_act_pos_obj = nn.CrossEntropyLoss(reduce=False)

        self._desc_prob = nn.CrossEntropyLoss(reduce=False)
        self._describer_obj = nn.CrossEntropyLoss()
        self._segmenter_obj = nn.BCEWithLogitsLoss()

        if FLAGS.gpu:
            for module in [
                    self._featurizer, self._policy, #self._flat_policy,
                    self._describer, self._segmenter, self._policy_act_obj,
                    self._policy_act_prob, self._policy_act_pos_obj,
                    self._desc_prob, self._describer_obj, self._segmenter_obj]:
                module.cuda()

        self._parser = TopDownParser(self, env, dataset)

        params = it.chain(
            self._featurizer.parameters(), self._policy.parameters(),
            self._describer.parameters(), self._segmenter.parameters())
        # TODO magic
        self._opt = optim.Adam(params, lr=0.001)

    def parse(self, seq_batch):
        return self._parser.parse(seq_batch)

    def act(self, step_batch, sample=True):
        rep, conv_rep = self._featurizer(step_batch.init_obs, step_batch.obs)
        act_logits, act_pos_logits, _ = self._policy(rep, conv_rep, step_batch)
        act_probs = self._policy._act_softmax(act_logits)
        act_probs = act_probs.data.cpu().numpy()
        act_pos_probs = self._policy._act_pos_softmax(act_pos_logits)
        act_pos_probs = act_pos_probs.data.cpu().numpy()
        out = []
        for i in range(act_probs.shape[0]):
            arow = act_probs[i, :]
            aprow = act_pos_probs[i, :]
            if sample:
                a = np.random.choice(arow.size, p=arow)
                ap = np.random.choice(aprow.size, p=aprow)
            else:
                a = arow.argmax()
                ap = aprow.argmax()
            # TODO size from elsewhere
            ap = np.unravel_index(ap, step_batch.init_obs.shape[2:])
            out.append((a, ap))
        return out

    def train_step(self, step_batch):
        feats, conv_feats = self._featurizer(step_batch.init_obs, step_batch.obs)
        act_logits, act_pos_logits, _ = self._policy(feats, conv_feats, step_batch)
        pol_loss = (
            self._policy_act_obj(act_logits, step_batch.act)
            + (self._policy_act_pos_obj(act_pos_logits, step_batch.act_pos)
                * step_batch.act_pos_mask).mean())

        seg_loss = self._segmenter_obj(self._segmenter(feats), step_batch.final)
        loss = pol_loss + seg_loss
        self._opt.zero_grad()
        loss.backward()
        self._opt.step()
        return Counter({'pol_loss': loss.data.cpu().numpy()[0]})

    def train_seq(self, seq_batch):
        feats, conv_feats = self._featurizer(seq_batch.init_obs, seq_batch.last_obs)
        _, desc_logits = self._describer(feats.unsqueeze(0), seq_batch.desc)
        n_tok, n_batch, n_pred = desc_logits.shape
        desc_loss = self._describer_obj(
            desc_logits.view(n_tok * n_batch, n_pred),
            seq_batch.desc_target.view(n_tok * n_batch))
        loss = desc_loss
        self._opt.zero_grad()
        loss.backward()
        self._opt.step()
        return Counter({'desc_loss': desc_loss.data.cpu().numpy()[0]})
