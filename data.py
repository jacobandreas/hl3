import ling
from misc import fakeprof, hlog, util

from collections import Counter, namedtuple
import gflags
import numpy as np
import os
import pickle
import torch
import torch.utils.data as torch_data
from torch.autograd import Variable
from tqdm import tqdm

FLAGS = gflags.FLAGS

class Batch(namedtuple('Batch', ['tasks', 'actions', 'features'])):
    pass

class SeqBatch(namedtuple('SeqBatch',
        ['desc', 'desc_target', 'init_obs', 'last_obs', 'all_obs', 'tasks'])):

    def cuda(self):
        cu_args = [a.cuda() if isinstance(a, Variable) else a for a in self]
        return SeqBatch(*cu_args)

    @classmethod
    def of(cls, batch, dataset):
        tasks, _, features = batch

        max_demo_len = max(f.shape[0] for f in features)
        n_feats = features[0].shape[1]

        desc = []
        init_obs = np.zeros((len(tasks), n_feats))
        last_obs = np.zeros((len(tasks), n_feats))
        all_obs = np.zeros((len(tasks), max_demo_len, n_feats))
        for task_id in range(len(tasks)):
            demo_len = features[task_id].shape[0]
            desc.append(tasks[task_id].desc)
            init_obs[task_id, :] = features[task_id][0]
            last_obs[task_id, :] = features[task_id][-1]
            all_obs[task_id, :demo_len, :] = features[task_id]

        desc_data, desc_target_data = load_desc_data(desc, dataset, target=True)
        init_obs_data = torch.FloatTensor(init_obs)
        last_obs_data = torch.FloatTensor(last_obs)
        all_obs_data = torch.FloatTensor(all_obs)
        out = SeqBatch(
            Variable(desc_data), Variable(desc_target_data),
            Variable(init_obs_data), Variable(last_obs_data),
            Variable(all_obs_data), tasks)

        if FLAGS.gpu:
            out = out.cuda()
        return out

class StepBatch(namedtuple('StepBatch',
        ['init_obs', 'obs', 'act', 'final', 'desc_in', 'desc_out_mask', 'desc_out', 'desc_out_target'])):

    def cuda(self):
        cu_args = [a.cuda() if isinstance(a, Variable) else a for a in self]
        return StepBatch(*cu_args)

    @classmethod
    def of(cls, batch, parses, dataset):
        tasks, actions, features = batch

        init_obs = []
        obs = []
        act = []
        final = []
        desc_in = []
        for _ in range(FLAGS.n_batch_steps):
            i_task = np.random.randint(len(tasks))
            i_step = np.random.randint(len(actions[i_task]))
            init_obs.append(features[i_task][0])
            obs.append(features[i_task][i_step])
            act.append(actions[i_task][i_step])
            final.append(i_step == len(actions[i_task])-1)
            desc_in.append(tasks[i_task].desc)

        init_obs_data = torch.FloatTensor(init_obs)
        obs_data = torch.FloatTensor(obs)
        act_data = torch.LongTensor(act)
        final_data = torch.FloatTensor(final)
        desc_in_data = load_desc_data(desc_in, dataset)
        out = StepBatch(
            Variable(init_obs_data), Variable(obs_data), Variable(act_data),
            Variable(final_data), Variable(desc_in_data), None, None, None)

        if FLAGS.gpu:
            out = out.cuda()
        return out

class DiskDataset(torch_data.Dataset):
    N_VAL = 30

    def __init__(self, cache_dir, n_batch, vocab, env,
            validation=False):
        self.cache_dir = cache_dir
        self.n_batch = n_batch
        self.vocab = vocab
        self.env = env
        self.validation = validation
        with open(os.path.join(cache_dir, 'actions.pkl'), 'rb') as action_f:
            self.actions = pickle.load(action_f)

    def __len__(self):
        if self.validation:
            return self.N_VAL * self.n_batch
        else:
            return len(self.actions) - self.N_VAL * self.n_batch

    def __getitem__(self, i):
        if self.validation:
            i += len(self.actions) - self.N_VAL * self.n_batch
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

def cache_dataset(env):
    os.mkdir(FLAGS.cache_dir)
    os.mkdir(os.path.join(FLAGS.cache_dir, 'features'))
    os.mkdir(os.path.join(FLAGS.cache_dir, 'tasks'))
    actions = []
    for i_task in tqdm(list(range(FLAGS.n_examples))):
        task = env.sample_task()
        demo = task.demonstration()
        t_actions = []
        features = []
        for s, a, s_ in demo:
            t_actions.append(a)
            features.append(s.features())

        actions.append(t_actions)

        with open(os.path.join(FLAGS.cache_dir, 'tasks', 'task%d.pkl' % i_task), 'wb') as task_f:
            pickle.dump(task, task_f)

        np.savez(
            os.path.join(FLAGS.cache_dir, 'features', 'path%d.npz' % i_task),
            np.asarray(features))

    with open(os.path.join(FLAGS.cache_dir, 'actions.pkl'), 'wb') as action_f:
        pickle.dump(actions, action_f)

def get_dataset(env):
    vocab = util.Index()
    with hlog.task('setup'):
        data = [env.sample_task() for _ in range(FLAGS.n_setup_examples)]
        vocab.index(ling.UNK)
        for datum in data:
            ling.tokenize(datum.desc, vocab, index=True)
        dcounts = Counter(datum.desc for datum in data)
        with hlog.task('setup_data_size'):
            hlog.log(len(data))
        with hlog.task('vocab_size'):
            hlog.log(len(vocab))

    if FLAGS.cache_dir is None:
        dataset = DynamicDataset(FLAGS.n_batch_examples, vocab, env)
        val_dataset = DynamicDataset(FLAGS.n_batch_examples, vocab, env)
    else:
        if not os.path.exists(FLAGS.cache_dir):
            cache_dataset(env)

        dataset = DiskDataset(
            FLAGS.cache_dir, FLAGS.n_batch_examples, vocab, env)
        val_dataset = DiskDataset(
            FLAGS.cache_dir, FLAGS.n_batch_examples, vocab, env,
            validation=True)

    val_interesting = None
    for i in range(len(val_dataset)):
        if val_dataset[i][0].desc == 'add a wood course':
            val_interesting = val_dataset[i]
            #break
    assert val_interesting is not None

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

def collate(items, dataset):
    tasks, actions, features = zip(*items)
    return Batch(tasks, actions, features)
