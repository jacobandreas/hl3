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

class Cudable(object):
    def cuda(self):
        cu_args = []
        for arg in self:
            if isinstance(arg, Variable):
                cu_args.append(arg.cuda())
            elif isinstance(arg, tuple):
                cu_args.append(tuple(
                    a.cuda() if isinstance(a, Variable) else a for a in arg))
            else:
                cu_args.append(arg)
        return type(self)(*cu_args)

class Batch(namedtuple('Batch', ['tasks', 'actions', 'obs'])):
    def length_filter(self, max_len):
        indices = [i for i in range(len(self.tasks)) 
            if len(self.actions[i]) < max_len]
        tasks = [self.tasks[i] for i in indices]
        actions = [self.actions[i] for i in indices]
        obs = tuple([opart[i] for i in indices] for opart in self.obs)
        kept = len(tasks)
        return Batch(tasks, actions, obs), kept

class SeqBatch(
        namedtuple('SeqBatch',
            ['obs', 'length', 'act', 'act_mask', 'desc', 'desc_tgt', 'tasks']),
        Cudable):

    @classmethod
    def of(cls, batch, dataset):
        tasks, actions, (state_obs, world_obs) = batch

        n_seqs = len(tasks)
        # shape examples for first sequence at first timestep
        max_len = max(s.shape[0] for s in state_obs)
        state_obs_shape = state_obs[0][0, ...].shape
        world_obs_shape = world_obs[0][0, ...].shape

        out_state_obs = np.zeros((n_seqs, max_len) + state_obs_shape)
        out_world_obs = np.zeros((n_seqs, max_len) + world_obs_shape)
        out_length = []
        out_act = []
        out_act_mask = []
        out_desc_raw = []
        for task_id in range(len(tasks)):
            so = state_obs[task_id]
            wo = world_obs[task_id]
            l = so.shape[0]
            out_state_obs[task_id, :l, ...] = state_obs[task_id][:l, ...]
            out_world_obs[task_id, :l, ...] = world_obs[task_id][:l, ...]
            out_length.append(l)
            a, a_mask = zip(*[dataset.ravel_action(a) for a in actions[task_id]])
            out_act.append(a)
            out_act_mask.append(a_mask)
            out_desc_raw.append(tasks[task_id].desc)

        out_state_obs = Variable(torch.FloatTensor(out_state_obs))
        out_world_obs = Variable(torch.FloatTensor(out_world_obs))
        out_desc, out_desc_tgt = load_desc_data(
            out_desc_raw, dataset, target=True)
        out_desc = Variable(out_desc)
        out_desc_tgt = Variable(out_desc_tgt)
        
        out_batch = SeqBatch(
            (out_state_obs, out_world_obs), out_length, out_act, out_act_mask,
            out_desc, out_desc_tgt, tasks)
        if FLAGS.gpu:
            out_batch = out_batch.cuda()
        return out_batch

    def init_obs(self):
        return tuple(torch.stack(
                [opart[i, 0, ...] for i in range(len(self.tasks))])
            for opart in self.obs)

    def last_obs(self):
        return tuple(torch.stack(
                [opart[i, t-1, ...] for i, t in enumerate(self.length)])
            for opart in self.obs)

class StepBatch(
        namedtuple('StepBatch', [
            'init_obs', 'obs', 'act', 'act_mask', 'final', 'desc_in',
            'desc_out', 'desc_out_tgt']),
        Cudable):

    @classmethod
    def for_states(cls, init_obs, obs, desc, dataset):
        init_state_obs, init_world_obs = zip(*init_obs)
        state_obs, world_obs = zip(*obs)

        init_state_obs = Variable(torch.FloatTensor(init_state_obs))
        init_world_obs = Variable(torch.FloatTensor(init_world_obs))
        state_obs = Variable(torch.FloatTensor(state_obs))
        world_obs = Variable(torch.FloatTensor(world_obs))

        desc_in = Variable(load_desc_data(desc, dataset))

        out_batch = StepBatch(
            (init_state_obs, init_world_obs), (state_obs, world_obs),
            None, None, None,
            desc_in,
            None, None)
        if FLAGS.gpu:
            out_batch = out_batch.cuda()
        return out_batch

    @classmethod
    def for_seq(
            cls, seq_batch, i_task, start, end, dataset, raw_desc=None, 
            tok_desc=None):
        obs = tuple(opart[i_task, start:end, ...] for opart in seq_batch.obs)
        init_obs = (opart[i_task, ...] for opart in seq_batch.init_obs())
        init_obs = tuple(io.expand_as(o) for io, o in zip(init_obs, obs))

        act = tuple(Variable(torch.LongTensor(a))
            for a in zip(*seq_batch.act[i_task][start:end]))
        act_mask = tuple(Variable(torch.FloatTensor(a))
            for a in zip(*seq_batch.act_mask[i_task][start:end]))

        if raw_desc is not None:
            desc_in = load_desc_data([raw_desc], dataset, tokenize=True)
        elif tok_desc is not None:
            desc_in = load_desc_data([tok_desc], dataset, tokenize=False)
        else:
            desc_in = load_desc_data(
                [seq_batch.tasks[i_desc].desc], dataset, tokenize=True)
        desc_in = desc_in.expand(-1, end-start, -1)
        desc_in = Variable(desc_in)

        out_batch = StepBatch(
            init_obs, obs, act, act_mask, None, desc_in, None, None)
        if FLAGS.gpu:
            out_batch = out_batch.cuda()
        return out_batch

    @classmethod
    def of(cls, batch, dataset, hier=False, parses=None):
        assert hier == (parses is not None)
        tasks, actions, (state_obs, world_obs) = batch

        n_seqs = len(tasks)
        n_steps = sum(len(acts) for acts in actions)
        def to_indices(step):
            counter = 0
            for task_id in range(n_seqs):
                seq_len = len(actions[task_id])
                if step - counter < seq_len:
                    return task_id, step - counter
                counter += seq_len
            assert False

        out_init_state_obs = []
        out_init_world_obs = []
        out_state_obs = []
        out_world_obs = []
        out_act = []
        out_act_mask = []
        out_final = []
        out_desc_in_raw = []
        out_desc_out_raw = []
        for _ in range(FLAGS.n_batch_steps):
            step = np.random.randint(n_steps)
            i_task, i_step = to_indices(step)
            out_init_state_obs.append(state_obs[i_task][0, ...])
            out_init_world_obs.append(world_obs[i_task][0, ...])
            out_state_obs.append(state_obs[i_task][i_step, ...])
            out_world_obs.append(world_obs[i_task][i_step, ...])

            if hier:
                task_parses = parses[i_task]
                desc = None
                final = i_step == len(actions[i_task]) - 1
                for parse in task_parses:
                    d, (start, end) = parse
                    if start <= i_step < end:
                        desc = d
                        break
                if desc is not None:
                    # TODO not here
                    a = (dataset.env.SAY, 0)
                    d = desc
                else:
                    # TODO yuck
                    a = (dataset.env.STOP if final else 0, 0)
                    d = []
                out_act.append(a)
                out_act_mask.append((1, 0))
                out_final.append(final)
                out_desc_in_raw.append(tasks[i_task].desc)
                out_desc_out_raw.append(d)
            else:
                a, a_mask = dataset.ravel_action(actions[i_task][i_step])
                out_act.append(a)
                out_act_mask.append(a_mask)
                out_final.append(i_step == len(actions[i_task]) - 1)
                out_desc_in_raw.append(tasks[i_task].desc)

        out_init_state_obs = Variable(torch.FloatTensor(out_init_state_obs))
        out_init_world_obs = Variable(torch.FloatTensor(out_init_world_obs))
        out_state_obs = Variable(torch.FloatTensor(out_state_obs))
        out_world_obs = Variable(torch.FloatTensor(out_world_obs))
        out_act = zip(*out_act)
        out_act = tuple(Variable(torch.LongTensor(a)) for a in out_act)
        out_act_mask = zip(*out_act_mask)
        out_act_mask = tuple(Variable(torch.FloatTensor(am))
            for am in out_act_mask)
        out_final = Variable(torch.FloatTensor(out_final))
        out_desc_in = Variable(load_desc_data(out_desc_in_raw, dataset))

        if len(out_desc_out_raw) == 0:
            out_desc_out = out_desc_out_tgt = None
        else:
            out_desc_out, out_desc_out_tgt = load_desc_data(
                out_desc_out_raw, dataset, target=True, tokenize=False)
            out_desc_out = Variable(out_desc_out)
            out_desc_out_tgt = Variable(out_desc_out_tgt)

        out_batch = StepBatch(
            (out_init_state_obs, out_init_world_obs),
            (out_state_obs, out_world_obs),
            out_act, out_act_mask, out_final, out_desc_in, 
            out_desc_out, out_desc_out_tgt)
        if FLAGS.gpu:
            out_batch = out_batch.cuda()
        return out_batch

class Hl3Dataset(object):
    def ravel_action(self, action):
        command, pos = action
        if pos is None:
            return (command, 0), (1, 0)
        pos = int(np.ravel_multi_index(pos, self.env.world_shape))
        return (command, pos), (1, 1)

    def unravel_action(self, action):
        command, pos = action
        return command, np.unravel_index(pos, self.env.world_shape)

    def render_desc(self, desc):
        return ' '.join(self.vocab.get(t) for t in desc[1:-1])

class DiskDataset(torch_data.Dataset, Hl3Dataset):
    def __init__(self, cache_dir, vocab, env):
        self.cache_dir = cache_dir
        self.vocab = vocab
        self.env = env
        with open(os.path.join(cache_dir, 'actions.pkl'), 'rb') as action_f:
            self.actions = pickle.load(action_f)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, i):
        with open(os.path.join(self.cache_dir, 'tasks', 'task%d.pkl' % i), 'rb') as task_f:
            task = pickle.load(task_f)
        state_obs = np.load(os.path.join(self.cache_dir, 'state_obs', 'path%d.npz' % i))['arr_0']
        world_obs = np.load(os.path.join(self.cache_dir, 'world_obs', 'path%d.npz' % i))['arr_0']

        return task, self.actions[i], (state_obs, world_obs)

class DynamicDataset(torch_data.Dataset, Hl3Dataset):
    def __init__(self, vocab, env):
        self.vocab = vocab
        self.env = env
        self._count = 0

    def __len__(self):
        return 1000

    def __getitem__(self, i):
        task = self.env.sample_task(self._count)
        self._count += 1
        demo = task.demonstration()
        task.validate(demo[-1][0])
        actions = [a for s, a, s_ in demo]
        all_obs = [s.obs() for s, a, s_ in demo]
        state_obs, world_obs = zip(*all_obs)
        return task, actions, (np.asarray(state_obs), np.asarray(world_obs))

def cache_dataset(env, cache_dir, n_examples, interesting=False):
    os.mkdir(cache_dir)
    os.mkdir(os.path.join(cache_dir, 'state_obs'))
    os.mkdir(os.path.join(cache_dir, 'world_obs'))
    os.mkdir(os.path.join(cache_dir, 'tasks'))
    actions = []
    for i_task in tqdm(list(range(n_examples))):
        task = env.sample_task(i_task, interesting=interesting)
        demo = task.demonstration()
        t_actions = []
        state_obs = []
        world_obs = []
        for s, a, s_ in demo:
            t_actions.append(a)
            s_o, w_o = s.obs()
            state_obs.append(s_o)
            world_obs.append(w_o)

        actions.append(t_actions)

        with open(os.path.join(
                cache_dir, 'tasks', 'task%d.pkl' % i_task), 'wb') \
                as task_f:
            pickle.dump(task, task_f)

        np.savez(
            os.path.join(cache_dir, 'state_obs', 'path%d.npz' % i_task),
            np.asarray(state_obs))

        np.savez(
            os.path.join(cache_dir, 'world_obs', 'path%d.npz' % i_task),
            np.asarray(world_obs))

    with open(os.path.join(cache_dir, 'actions.pkl'), 'wb') as action_f:
        pickle.dump(actions, action_f)

def get_dataset(env):
    vocab = env.vocab

    assert (FLAGS.cache_dir is None) == (FLAGS.val_cache_dir is None)
    skip_cache = FLAGS.cache_dir is None

    if skip_cache is None:
        dataset = DynamicDataset(vocab, env)
        val_dataset = DynamicDataset(vocab, env)
    else:
        if not os.path.exists(FLAGS.cache_dir):
            cache_dataset(env, FLAGS.cache_dir, FLAGS.n_examples)
        dataset = DiskDataset(FLAGS.cache_dir, vocab, env)

        if not os.path.exists(FLAGS.val_cache_dir):
            cache_dataset(
                env, FLAGS.val_cache_dir, FLAGS.n_val_examples,
                interesting=True)
        val_dataset = DiskDataset(FLAGS.val_cache_dir, vocab, env)

    return dataset, val_dataset

def load_desc_data(descs, dataset, target=False, tokenize=True):
    if tokenize:
        assert all(isinstance(d, str) for d in descs)
        proc_descs = [ling.tokenize(d, dataset.vocab) for d in descs]
    else:
        proc_descs = descs
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
            target_data[i_tok-1, i_desc] = int(tok)
    return desc_data, target_data


def collate(items, dataset):
    tasks, actions, obs = zip(*items)
    # every observation is a tuple
    obs = list(zip(*obs))
    return Batch(tasks, actions, obs)
