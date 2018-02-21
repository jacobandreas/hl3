import ling

import itertools as it
import torch
from torch import nn
from torch.autograd import Variable

class StateFeaturizer(nn.Module):
    N_HIDDEN = 256

    def __init__(self, env, dataset):
        super(StateFeaturizer, self).__init__()
        n_obs = env.n_features
        self.layers = nn.Sequential(
            nn.Linear(n_obs, self.N_HIDDEN),
            nn.ReLU(),
            nn.Linear(self.N_HIDDEN, self.N_HIDDEN))

    def forward(self, obs):
        return self.layers(obs)

class Policy(nn.Module):
    # TODO auto
    N_WORDVEC = 64
    N_HIDDEN = 256

    def __init__(self, env, dataset):
        super(Policy, self).__init__()
        self._n_actions = env.n_actions
        self.embed = nn.Linear(len(dataset.vocab), self.N_WORDVEC)
        self.rnn = nn.GRU(
            input_size=self.N_WORDVEC, hidden_size=self.N_HIDDEN, num_layers=1)

        self.predict = nn.Bilinear(self.N_HIDDEN, StateFeaturizer.N_HIDDEN, self._n_actions)
        self.log_softmax = nn.LogSoftmax(dim=1)

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
            actions.append(np.random.choice(self._n_actions, p=row))
        return actions

class Splitter(nn.Module):
    def __init__(self, env, dataset):
        super(Splitter, self).__init__()
        self.predict = nn.Linear(StateFeaturizer.N_HIDDEN, 1)

    def forward(self, features, batch):
        scores = self.predict(features).squeeze(2)
        return scores

class Decoder(nn.Module):
    N_HIDDEN = 256

    def __init__(self, dataset, start_sym, stop_sym):
        super(Decoder, self).__init__()
        self.vocab = dataset.vocab
        self.start_sym = start_sym
        self.stop_sym = stop_sym
        # TODO switch to word embeddings / sparse?
        self.emb = nn.Linear(len(self.vocab), self.N_HIDDEN)
        self.rnn = nn.GRU(
            input_size=self.N_HIDDEN, hidden_size=self.N_HIDDEN, num_layers=1)
        self.out = nn.Linear(self.N_HIDDEN, len(self.vocab))
        self.log_softmax = nn.LogSoftmax(dim=1)

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
                choice = probs.argmax()
                new_inp.append(choice)
                if not done[i]:
                    out[i].append(choice)
                done[i] = done[i] or choice == self.vocab[self.stop_sym]
            state = new_state
            inp = new_inp
        return out

class Model(object):
    def __init__(self, env, dataset):
        self.featurizer = StateFeaturizer(env, dataset).cuda()
        self.policy = Policy(env, dataset).cuda()
        self.splitter = Splitter(env, dataset).cuda()
        self.describer = Decoder(dataset, ling.START, ling.STOP).cuda()

        self.policy_obj = nn.NLLLoss().cuda()
        self.policy_prob = nn.NLLLoss(size_average=False)
        self.describer_obj = nn.NLLLoss().cuda()

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

        ## # splitter
        ## seg_features = self.featurizer(seg_batch.obs)
        ## split_scores = self.splitter(seg_features, seg_batch)
        ## # TODO make a module
        ## baselines = split_scores.gather(1, seg_batch.final[:, np.newaxis])
        ## baselined_scores = split_scores - baselines
        ## augmented_scores = baselined_scores + seg_batch.loss
        ## margin_losses = nn.functional.relu(augmented_scores)
        ## bottoms, _ = split_scores.min(1, keepdim=True)
        ## bottomed_scores = split_scores - bottoms
        ## split_loss = margin_losses.mean() + .1 * bottomed_scores.mean()
        ## split_pred_final = split_scores.data.cpu().numpy().argmax(axis=1)
        ## split_acc = (split_pred_final == seg_batch.final.data.cpu().numpy()).mean()
        split_loss = 0

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
        #split_score = split_loss.data.cpu().numpy()[0]
        desc_score = desc_loss.data.cpu().numpy()[0]
        return Counter({
            'pol_nll': pol_score,
            #'split_loss': split_score,
            #'split_acc': split_acc,
            'desc_nll': desc_score,
        })

    def act(self, pol_batch):
        features = self.featurizer(pol_batch.obs)
        return self.policy.act(features, pol_batch)

    def parse(self, seg_batch):
        assert len(seg_batch.tasks) == 1
        demo = seg_batch.tasks[0].demonstration()

        feature_cache = {}
        for i in range(len(demo)):
            state_before, _, _ = demo[i]
            for j in range(i, len(demo)):
                state, _, _ = demo[j]
                state = state.with_init(state_before)
                feature_cache[i, j] = state.features()

        def score_segment(i, j, desc):
            hot_desc = np.zeros((len(desc), j-i+1, len(self.dataset.vocab)))
            for p, t in enumerate(desc):
                hot_desc[p, :, t] = 1

            features = [feature_cache[i, k] for k in range(i, j+1)]
            actions = [demo[k][1] for k in range(i, j)] + [ENV.STOP]

            batch = data.PolicyBatch(
                Variable(torch.FloatTensor(hot_desc)).cuda(),
                None,
                Variable(torch.FloatTensor(features)).cuda(),
                Variable(torch.LongTensor(actions)).cuda(),
                None)
            rep = self.featurizer(batch.obs)
            preds = self.policy(rep, batch)
            score = self.policy_prob(preds, batch.act)
            return score

        def propose_desc(i, j, n):
            feats = [feature_cache[i, j] for _ in range(n)]
            feats_var = Variable(torch.FloatTensor(feats)).cuda()
            reps = self.featurizer(feats_var)
            descs = self.describer.decode(reps[np.newaxis, ...])
            return descs

        # s a s a s a s a s a s a s a s x s
        # 0   1   2   3   4   5   6   7   8
        # [           |               ]
        # 

        def render(desc):
            return ' '.join(self.dataset.vocab.get(t) for t in desc)

        descs = []
        scores = []
        full_scores = []
        #for k in range(1, len(demo)):
        #for k in range(16, 18):
        #splits = [17]
        splits = list(range(1, len(demo)-1))
        assert len(demo) >= 3
        for k in splits:
            desc1, = propose_desc(0, k, 1)
            desc2, = propose_desc(k, len(demo)-1, 1)
            descs.append((render(desc1), render(desc2)))
            s1 = score_segment(0, k, desc1).data.cpu().numpy()[0]
            s2 = score_segment(k, len(demo)-1, desc2).data.cpu().numpy()[0]
            score = s1 + s2
            scores.append(score)
            full_scores.append((s1, s2))
        print(scores)
        i_split = np.asarray(scores).argmin()
        split = splits[i_split]
        print(0, split, len(demo)-1, descs[i_split])
        actions = [a for s, a, s in demo]
        print(actions[:split], actions[split:])
        print(full_scores[splits.index(17)], full_scores[i_split])


    #def parse(self, seg_batch):
    #    assert len(seg_batch.tasks) == 1
    #    demo = seg_batch.tasks[0].demonstration()
    #    actions = [a for s, a, s_ in demo]

    #    # TODO probably move to data
    #    feature_cache = {}
    #    def featurize_from(lo):
    #        if lo in feature_cache:
    #            return feature_cache[lo]
    #        state_before = demo[lo][0]
    #        features = [None] * len(demo)
    #        for i in range(lo, len(demo)):
    #            state_after = demo[i][0]
    #            state = state_after.with_init(state_before)
    #            features[i] = state.features()
    #        feature_cache[lo] = features
    #        return features

    #    #feature_cache = {}
    #    #def desc_split(lo, hi, depth):
    #    #    if lo not in feature_cache:
    #    #        feature_cache[lo] = featurize_from(lo)
    #    #    features = feature_cache[lo][lo:hi+1]
    #    #    split_batch = data.SegmentBatch(
    #    #        None, None,
    #    #        Variable(torch.FloatTensor([features])).cuda(),
    #    #        Variable(torch.FloatTensor([features[-1]])).cuda(),
    #    #        None, None, None)
    #    #    last_obs_feats = self.featurizer(split_batch.last_obs)

    #    #    print(features[-1])

    #    #    desc = self.describer.decode(last_obs_feats[np.newaxis, ...])[0]
    #    #    desc = ' '.join(self.dataset.vocab.get(t) for t in desc)
    #    #    if depth == 0:
    #    #        return ((lo, hi), desc)

    #    #    obs_feats = self.featurizer(split_batch.obs)
    #    #    split_scores = self.splitter(obs_feats, split_batch)
    #    #    print(split_scores)
    #    #    split = lo + split_scores.data.cpu().numpy().ravel()[:-1].argmax()
    #    #    return (
    #    #        (lo, hi),
    #    #        desc,
    #    #        desc_split(lo, split, depth-1),
    #    #        desc_split(split, hi, depth-1))

    #    def desc_split(lo, hi, depth):
    #        assert hi > lo
    #        features_before = featurize_from(lo)

    #        candidates = []

    #        for split in range(lo + 1, hi - 1):
    #            features_after = featurize_from(split)

    #            before_batch = data.SegmentBatch(
    #                None, None, None,
    #                Variable(torch.FloatTensor([features_before[split]])).cuda(),
    #                None, None, None)

    #            after_batch = data.SegmentBatch(
    #                None, None, None,
    #                Variable(torch.FloatTensor([features_after[hi]])).cuda(),
    #                None, None, None)

    #            obs_feats_before = self.featurizer(before_batch.last_obs)
    #            obs_feats_after = self.featurizer(after_batch.last_obs)

    #            desc_before = self.describer.decode(obs_feats_before[np.newaxis, ...])[0]
    #            desc_after = self.describer.decode(obs_feats_after[np.newaxis, ...])[0]
    #            hot_desc_before = np.zeros((len(desc_before), split-lo+1, len(self.dataset.vocab)))
    #            hot_desc_after = np.zeros((len(desc_after), hi-split+1, len(self.dataset.vocab)))
    #            for i in range(len(desc_before)):
    #                hot_desc_before[i, :, desc_before[i]] = 1
    #            for i in range(len(desc_after)):
    #                hot_desc_after[i, :, desc_after[i]] = 1

    #            before_pol_batch = data.PolicyBatch(
    #                Variable(torch.FloatTensor(hot_desc_before)).cuda(),
    #                None,
    #                Variable(torch.FloatTensor(features_before[lo:split+1])).cuda(),
    #                Variable(torch.LongTensor(actions[lo:split] + [self.dataset.env.STOP])).cuda(),
    #                None)

    #            after_pol_batch = data.PolicyBatch(
    #                Variable(torch.FloatTensor(hot_desc_after)).cuda(),
    #                None,
    #                Variable(torch.FloatTensor(features_after[split:hi+1])).cuda(),
    #                Variable(torch.LongTensor(actions[split:hi] + [self.dataset.env.STOP])).cuda(),
    #                None)

    #            before_feats = self.featurizer(before_pol_batch.obs)
    #            after_feats = self.featurizer(after_pol_batch.obs)

    #            #print('bad')
    #            #print(before_feats.shape)
    #            #print(before_pol_batch.desc.shape)
    #            #print(before_pol_batch.act.shape)
    #            #print(after_feats.shape)
    #            #print(after_pol_batch.desc.shape)
    #            #print(after_pol_batch.act.shape)
    #            before_preds = self.policy(before_feats, before_pol_batch)
    #            after_preds = self.policy(after_feats, after_pol_batch)

    #            before_loss = self.policy_obj(before_preds, before_pol_batch.act)
    #            after_loss = self.policy_obj(after_preds, after_pol_batch.act)
    #            candidates.append((
    #                before_loss.data.cpu().numpy()[0] +
    #                    after_loss.data.cpu().numpy()[0],
    #                (lo, split, hi),
    #                ' '.join(self.dataset.vocab.get(t) for t in desc_before),
    #                ' '.join(self.dataset.vocab.get(t) for t in desc_after)
    #            ))
    #            #print(before_loss, after_loss)
    #        score, (lo, split, hi), desc1, desc2 = max(candidates, key=lambda x: x[0])
    #        return [((lo, split), desc1), ((split, hi), desc2)]

    #    return desc_split(0, len(demo)-1, 1)

