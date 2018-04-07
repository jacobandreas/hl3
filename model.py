import data
import ling
from misc.util import unwrap
from parser import TopDownParser

from collections import Counter
import gflags
import itertools as it
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable

FLAGS = gflags.FLAGS
gflags.DEFINE_integer('hidden_size', 256, 'common size of hidden states')
gflags.DEFINE_integer('filter_size', 64, 'number of intermediate conv filters')
gflags.DEFINE_integer('conv_window', 5, 'size of convolutional window')
gflags.DEFINE_integer('wordvec_size', 64, 'word vector size')

class WorldFeaturizer(nn.Module):
    def __init__(self, env, dataset):
        hid = FLAGS.hidden_size
        chid1 = FLAGS.filter_size
        chid2 = FLAGS.hidden_size
        k1 = FLAGS.conv_window
        super().__init__()
        self._conv = nn.Sequential(
            nn.Conv3d(2*env.n_world_obs, chid1, k1, padding=k1//2),
            nn.ReLU(),
            nn.Conv3d(chid1, chid2, 1))
        self._mlp = nn.Sequential(
            nn.Linear(2*env.n_state_obs, hid),
            nn.ReLU(),
            nn.Linear(hid, hid))

    def forward(self, init_obs, obs, skip_world=False):
        init_state_obs, init_world_obs = init_obs
        state_obs, world_obs = obs
        in_state_obs = torch.cat((state_obs, state_obs-init_state_obs), 1)
        state_feats = self._mlp(in_state_obs)
        if skip_world:
            return state_feats, None
        in_world_obs = torch.cat((world_obs, world_obs-init_world_obs), 1)
        world_feats = self._conv(in_world_obs)
        return state_feats, world_feats

class Policy(nn.Module):
    def __init__(self, env, dataset):
        wvec = FLAGS.wordvec_size
        hid = FLAGS.hidden_size
        super().__init__()
        self._n_actions = env.n_actions
        self._embed = nn.Linear(len(dataset.vocab), wvec)
        self._encoder = nn.GRU(input_size=wvec, hidden_size=hid, num_layers=1)
        self._decoder = Decoder(dataset.vocab, ling.START, ling.STOP)
        self._predict_act = nn.Bilinear(hid, hid, self._n_actions)
        self._act_softmax = nn.Softmax(dim=1)
        self._act_pos_softmax = nn.Softmax(dim=1)

    def forward(self, feats, batch):
        state_feats, world_feats = feats
        emb = self._embed(batch.desc_in)
        _, enc = self._encoder(emb)
        enc = enc.squeeze(0)
        act_logits = self._predict_act(enc, state_feats)

        n_batch, n_feats, _, _, _ = world_feats.shape
        tile_enc = enc.view(n_batch, n_feats, 1, 1, 1).expand_as(world_feats)
        act_pos_logits = (world_feats * tile_enc).sum(1).view(n_batch, -1)

        if batch.desc_out is None:
            return (act_logits, act_pos_logits), None
        else:
            comb_feats = (state_feats + enc).unsqueeze(0)
            _, desc_logits = self._decoder(comb_feats, batch.desc_out)
            return (act_logits, act_pos_logits), desc_logits

    def decode(self, feats, batch):
        state_feats, world_feats = feats
        emb = self._embed(batch.desc_in)
        _, enc = self._encoder(emb)
        enc = enc.squeeze(0)
        comb_feats = (state_feats + enc).unsqueeze(0)
        return self._decoder.decode(comb_feats, FLAGS.parse_desc_max_len)

class Segmenter(nn.Module):
    def __init__(self, env, dataset):
        super().__init__()
        self._predict = nn.Linear(FLAGS.hidden_size, 1)

    def forward(self, feats):
        state_feats, _ = feats
        return self._predict(state_feats).squeeze(-1)

class Decoder(nn.Module):
    def __init__(self, vocab, start_sym, stop_sym):
        hid = FLAGS.hidden_size
        super().__init__()
        self._vocab = vocab
        self._start_id = vocab[start_sym]
        self._stop_id = vocab[stop_sym]

        self._embed = nn.Linear(len(vocab), hid)
        self._rnn = nn.GRU(input_size=hid, hidden_size=hid, num_layers=1)
        self._predict = nn.Linear(hid, len(vocab))
        self._softmax = nn.Softmax(dim=1)

    def forward(self, state, inp):
        emb = self._embed(inp)
        rep, enc = self._rnn(emb, state)
        logits = self._predict(rep)
        return enc, logits

    @profile
    def decode(self, init_state, max_len, sample=False):
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
                if sample:
                    tok = np.random.choice(row.size, p=row)
                else:
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

class Model(nn.Module):
    def __init__(self, env, dataset):
        super().__init__()
        self._env = env
        self._dataset = dataset

        self._featurizer = WorldFeaturizer(env, dataset)
        self._flat_policy = Policy(env, dataset)
        self._hier_policy = Policy(env, dataset)
        self._describer = Decoder(dataset.vocab, ling.START, ling.STOP)
        self._segmenter = Segmenter(env, dataset)

        self._policy_act_logprob = nn.CrossEntropyLoss(reduce=False)
        self._policy_act_pos_logprob = nn.CrossEntropyLoss(reduce=False)
        self._policy_desc_logprob = nn.CrossEntropyLoss(reduce=False)

        self._desc_logprob = nn.CrossEntropyLoss(reduce=False)
        self._describer_obj = nn.CrossEntropyLoss()
        self._segmenter_obj = nn.BCEWithLogitsLoss()

        self._parser = TopDownParser(self, env, dataset)

        self._step_opt = optim.Adam(it.chain(
            self._featurizer.parameters(), self._flat_policy.parameters(),
            self._segmenter.parameters()))
        self._seq_opt = optim.Adam(it.chain(
            self._featurizer.parameters(), self._describer.parameters()))
        self._hier_opt = optim.Adam(self._hier_policy.parameters())

    def parse(self, seq_batch):
        return self._parser.parse(seq_batch)

    def act(self, step_batch, sample=True):
        feats = self._featurizer(step_batch.init_obs, step_batch.obs)
        (act_logits, act_pos_logits), _ = self._flat_policy(feats, step_batch)
        act_probs = unwrap(self._flat_policy._act_softmax(act_logits))
        act_pos_probs = unwrap(self._flat_policy._act_pos_softmax(act_pos_logits))
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
            a, ap = self._dataset.unravel_action((a, ap))
            out.append((a, ap))
        #print(out)
        return out

    def act_hier(self, step_batch, sample=True, feats=None):
        if feats is None:
            feats = self._featurizer(step_batch.init_obs, step_batch.obs)
        (act_logits, _), _= self._hier_policy(feats, step_batch)
        descs = self._hier_policy.decode(feats, step_batch)
        act_probs = unwrap(self._hier_policy._act_softmax(act_logits))

        top_actions = []
        for i in range(act_probs.shape[0]):
            arow = act_probs[i, :]
            if sample:
                a = np.random.choice(arow.size, p=arow)
            else:
                a = arow.argmax()
            top_actions.append(a)

        #print()
        #print(top_actions)
        #print([a == self._env.SAY for a in top_actions])
        
        ### # TODO modify init obs
        ### next_descs = []
        ### for i, a in enumerate(top_actions):
        ###     if a == self._env.SAY:
        ###         next_descs.append(Variable(data.load_desc_data(
        ###             descs[i:i+1], self._dataset, tokenize=False)))
        ###     else:
        ###         next_descs.append(step_batch.desc_in[:, i:i+1, :])
        ### # TODO not here
        ### next_descs = torch.cat([d.cuda() for d in next_descs], dim=1)

        next_descs = []
        for i, a in enumerate(top_actions):
            if a == self._env.SAY:
                next_descs.append(descs[i])
            else:
                next_descs.append(step_batch.desc[i])
        next_descs = Variable(
            data.load_desc_data(next_descs, self._dataset, tokenize=False))
        flat_batch = step_batch._replace(desc_in=next_descs).cuda()
        return self.act(flat_batch, sample=sample)

    def _logprob_of(self, policy, feats, step_batch):
        (act_logits, act_pos_logits), desc_out_logits = policy(feats, step_batch)
        act_targets, act_pos_targets = step_batch.act
        act_mask, act_pos_mask = step_batch.act_mask
        # TODO put some of this into Policy
        act_lp = (
            self._policy_act_logprob(act_logits, act_targets)
            * act_mask)
        act_pos_lp = (
            self._policy_act_pos_logprob(act_pos_logits, act_pos_targets) 
            * act_pos_mask)
        desc_lp = 0
        if desc_out_logits is not None:
            n_tok, n_batch, n_pred = desc_out_logits.shape
            desc_lp = self._policy_desc_logprob(
                desc_out_logits.view(n_tok * n_batch, n_pred),
                step_batch.desc_out_tgt.view(n_tok * n_batch)).mean()
        return act_lp + act_pos_lp + desc_lp

    def train_step(self, step_batch):
        feats = self._featurizer(step_batch.init_obs, step_batch.obs)
        pol_loss = self._logprob_of(self._flat_policy, feats, step_batch).mean()
        seg_loss = self._segmenter_obj(self._segmenter(feats), step_batch.final)
        loss = pol_loss + seg_loss
        self._step_opt.zero_grad()
        loss.backward()
        self._step_opt.step()
        return Counter({'pol_loss': unwrap(loss)[0]})

    def train_seq(self, seq_batch):
        state_feats, _ = self._featurizer(
            seq_batch.init_obs(), seq_batch.last_obs(), skip_world=True)
        _, desc_logits = self._describer(state_feats.unsqueeze(0), seq_batch.desc)
        n_tok, n_batch, n_pred = desc_logits.shape
        desc_loss = self._describer_obj(
            desc_logits.view(n_tok * n_batch, n_pred),
            seq_batch.desc_tgt.view(n_tok * n_batch))
        loss = desc_loss
        self._seq_opt.zero_grad()
        loss.backward()
        self._seq_opt.step()
        return Counter({'desc_loss': unwrap(loss)[0]})

    def train_hier(self, step_batch):
        feats = self._featurizer(step_batch.init_obs, step_batch.obs)
        loss = self._logprob_of(self._hier_policy, feats, step_batch).mean()
        self._hier_opt.zero_grad()
        loss.backward()
        self._hier_opt.step()
        return Counter({'hier_loss': unwrap(loss)[0]})
