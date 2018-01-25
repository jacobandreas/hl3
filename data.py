import ling
from misc import fakeprof, hlog, util

from collections import Counter, namedtuple
import numpy as np
import os
import pickle
import torch
import torch.utils.data as torch_data
from torch.autograd import Variable

class PolicyBatch(namedtuple('PolicyBatch', ['desc', 'desc_target', 'obs', 'act', 'done'])):
    def cuda(self):
        return PolicyBatch(
            self.desc.cuda(), self.desc_target.cuda(), self.obs.cuda(),
            self.act.cuda(), self.done.cuda())

class SegmentBatch(namedtuple('SegmentBatch', ['desc', 'desc_target', 'obs', 'last_obs', 'final', 'loss', 'tasks'])):
    def cuda(self):
        return SegmentBatch(
            self.desc.cuda(), self.desc_target.cuda(), self.obs.cuda(),
            self.last_obs.cuda(), self.final.cuda(), self.loss.cuda(),
            self.tasks)

START = '<s>'
STOP = '</s>'

class DiskDataset(torch_data.Dataset):
    def __init__(self, cache_dir, n_batch, vocab, act_feat_index, env,
            validation=False):
        self.cache_dir = cache_dir
        self.n_batch = n_batch
        self.vocab = vocab
        self.act_feat_index = act_feat_index
        self.env = env
        self.validation = validation
        with open(os.path.join(cache_dir, 'actions.pkl'), 'rb') as action_f:
            self.actions = pickle.load(action_f)

    def __len__(self):
        if self.validation:
            return 2 * self.n_batch
        else:
            return len(self.actions) - 2 * self.n_batch

    def __getitem__(self, i):
        if self.validation:
            i += len(self.actions) - 2 * self.n_batch
        with open(os.path.join(self.cache_dir, 'tasks', 'task%d.pkl' % i), 'rb') as task_f:
            task = pickle.load(task_f)
        features = np.load(os.path.join(self.cache_dir, 'features', 'path%d.npz' % i))['arr_0']
        return task, self.actions[i], features

class DynamicDataset(torch_data.Dataset):
    def __init__(self, n_batch, vocab, env):
        self.n_batch = n_batch
        self.vocab = vocab
        self.env = env

    def __len__(self):
        return 1000

    def __getitem__(self, i):
        task = self.env.sample_task()
        demo = task.demonstration()
        actions = [a for s, a, s_ in demo]
        features = np.asarray([s.features() for s, a, s_ in demo])
        return task, actions, features

def cache_dataset(env, config):
    os.mkdir(CACHE_DIR)
    os.mkdir(os.path.join(config.CACHE_DIR, 'features'))
    os.mkdir(os.path.join(config.CACHE_DIR, 'tasks'))
    actions = []
    for i_task in tqdm(list(range(config.N_EXAMPLES))):
        task = env.sample_task()
        demo = task.demonstration()
        t_actions = []
        features = []
        for s, a, s_ in demo:
            t_actions.append(a)
            features.append(s.features())

        actions.append(t_actions)

        with open(os.path.join(config.CACHE_DIR, 'tasks', 'task%d.pkl' % i_task), 'wb') as task_f:
            pickle.dump(task, task_f)

        np.savez(
            os.path.join(config.CACHE_DIR, 'features', 'path%d.npz' % i_task),
            np.asarray(features))

    with open(os.path.join(config.CACHE_DIR, 'actions.pkl'), 'wb') as action_f:
        pickle.dump(actions, action_f)

def get_dataset(env, config):
    vocab = util.Index()
    # TODO remove
    act_feat_index = util.Index()
    with hlog.task('setup'):
        data = [env.sample_task() for _ in range(config.N_SETUP_EXAMPLES)]
        vocab.index(ling.UNK)
        for datum in data:
            ling.tokenize(datum.desc, vocab, index=True)
        dcounts = Counter(datum.desc for datum in data)
        with hlog.task('setup_data_size'):
            hlog.log(len(data))
        with hlog.task('vocab_size'):
            hlog.log(len(vocab))

        act_feat_index.index(START)
        act_feat_index.index(STOP)
        for action in list(range(env.n_actions)) + [START, STOP]:
            act_feat_index.index((action,))
            for a2 in list(range(env.n_actions)) + [STOP]:
                act_feat_index.index((action, a2))

    if config.CACHE_DIR is None:
        dataset = DynamicDataset(config.N_BATCH_EXAMPLES, vocab, env)
        val_dataset = DynamicDataset(config.N_BATCH_EXAMPLES, vocab, env)
    else:
        if not os.path.exists(config.CACHE_DIR):
            cache_dataset(env, config)

        dataset = DiskDataset(config.CACHE_DIR, config.N_BATCH_EXAMPLES, vocab,
            act_feat_index, env)
        val_dataset = DiskDataset(config.CACHE_DIR, config.N_BATCH_EXAMPLES, vocab,
            act_feat_index, env, validation=True)

    val_interesting = val_dataset[14]
    assert val_interesting[0].desc == 'add a blue course'

    #for i in range(100):
    #    v = val_dataset[i]
    #    if v[0].desc == 'southeast':
    #        val_interesting = v
    #        break
    #assert val_interesting[0].desc == 'southeast'

    return (dataset, val_dataset, val_interesting)

def load_desc_data(descs, dataset, target=False):
    proc_descs = [ling.tokenize(d, dataset.vocab) for d in descs]
    max_desc_len = max(len(d) for d in proc_descs)
    desc_data = torch.zeros(max_desc_len, len(descs), len(dataset.vocab))
    for i_desc, desc in enumerate(proc_descs):
        for i_tok, tok in enumerate(desc):
            desc_data[i_tok, i_desc, tok] = 1
    if not target:
        return desc_data
    target_data = torch.LongTensor(max_desc_len, len(descs)).zero_()
    for i_desc, desc in enumerate(proc_descs):
        for i_tok, tok in enumerate(desc):
            if i_tok == 0: continue
            target_data[i_tok-1, i_desc] = tok
    return desc_data, target_data

def collate(items, dataset, config):
    tasks, actions, features = zip(*items)
    return (
        load_policy_batch(tasks, actions, features, dataset, config),
        load_segment_batch(tasks, actions, features, dataset, config))

@profile
def load_policy_batch(tasks, actions, features, dataset, config):
    desc = []
    act = []
    obs = []
    done = []
    for _ in range(config.N_BATCH_STEPS):
        task_id = np.random.randint(len(tasks))
        step_id = np.random.randint(len(actions[task_id]))
        desc.append(tasks[task_id].desc)
        obs.append(features[task_id][step_id])
        act.append(actions[task_id][step_id])
        done.append(int(step_id == len(actions[task_id]) - 1))

    desc_data, desc_target_data = load_desc_data(desc, dataset, target=True)
    obs_data = torch.FloatTensor(obs)
    act_data = torch.LongTensor(act)
    done_data = torch.LongTensor(done)
    return PolicyBatch(
        Variable(desc_data),
        Variable(desc_target_data),
        Variable(obs_data),
        Variable(act_data),
        Variable(done_data))

@profile
def load_segment_batch(tasks, actions, features, dataset, config):
    max_len = max(f.shape[0] for f in features)
    n_feats = features[0].shape[1]
    obs = np.zeros((len(tasks), max_len, n_feats))
    last_obs = np.zeros((len(tasks), n_feats))
    loss = np.ones((len(tasks), max_len))
    final = []
    desc = []
    for task_id in range(len(tasks)):
        d_len = features[task_id].shape[0]
        obs[task_id, :d_len, :] = features[task_id]
        last_obs[task_id, :] = features[task_id][-1]
        assert d_len > 0
        final.append(d_len-1)
        #loss_label[task_id, d_len-1] = 1
        loss[task_id, d_len-1] = 0
        desc.append(tasks[task_id].desc)

    desc_data, desc_target_data = load_desc_data(desc, dataset, target=True)
    obs_data = torch.FloatTensor(obs)
    last_obs_data = torch.FloatTensor(last_obs)
    final_data = torch.LongTensor(final)
    loss_data = torch.FloatTensor(loss)
    return SegmentBatch(
        Variable(desc_data),
        Variable(desc_target_data),
        Variable(obs_data),
        Variable(last_obs_data),
        Variable(final_data),
        Variable(loss_data),
        tasks)

#def collate_segment(items, dataset, config):
#    tasks, actions, _ = zip(*items)
#    return load_segment_batch(tasks, actions, dataset, config)

#@profile
#def load_segment_batch(tasks, actions, dataset, config):
#
#    act_features = torch.zeros(config.N_BATCH_STEPS, len(dataset.act_feat_index))
#    label = []
#
#    descs = []
#    for i_datum in range(config.N_BATCH_STEPS):
#        i1, i2 = np.random.randint(len(actions), size=(2,))
#        seg1 = actions[i1]
#        seg2 = actions[i2]
#        combined = seg1 + seg2
#
#        rand = np.random.random()
#        if rand < 0.9:
#            pick_len = np.random.randint(1, len(combined))
#            pick_start = np.random.randint(len(combined)-1)
#        elif rand < 0.95:
#            pick_len = len(seg1)
#            pick_start = 0
#        else:
#            pick_len = len(seg2)
#            pick_start = len(seg1)
#        picked = combined[pick_start:pick_start+pick_len]
#        picked = [START] + picked + [STOP]
#
#        for i_a in range(len(picked)):
#            a1 = picked[i_a]
#            f1 = dataset.act_feat_index[(a1,)]
#            assert f1 is not None
#            act_features[i_datum, f1] += 1
#            if i_a < len(picked) - 1:
#                a2 = picked[i_a+1]
#                f2 = dataset.act_feat_index[(a1, a2)]
#                assert f2 is not None
#                act_features[i_datum, f2] += 1
#        act_features[i_datum, :] /= sum(act_features[i_datum, :])
#
#        if pick_start == 0 and pick_len == len(seg1):
#            label.append(1)
#            descs.append(tasks[i1].desc)
#        elif pick_start == len(seg1) and pick_len == len(seg2):
#            label.append(1)
#            descs.append(tasks[i2].desc)
#        else:
#            label.append(0)
#            descs.append("")
#
#    desc_data, desc_target_data = load_desc_data(descs, dataset, target=True)
#
#    return SegmentBatch(
#        torch.autograd.Variable(desc_data),
#        torch.autograd.Variable(desc_target_data),
#        torch.autograd.Variable(act_features),
#        torch.autograd.Variable(torch.LongTensor(label)))
#
#    #return SegmentBatch(None, None, None)
