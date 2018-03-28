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

class Batch(namedtuple('Batch', ['tasks', 'actions', 'state_obs', 'world_obs'])):
    def length_filter(self, max_len):
        indices = [i for i in range(len(self.tasks)) if len(self.actions[i]) < max_len]
        tasks = [self.tasks[i] for i in indices]
        actions = [self.actions[i] for i in indices]
        state_obs = [self.state_obs[i] for i in indices]
        world_obs = [self.world_obs[i] for i in indices]
        kept = len(tasks)
        return Batch(tasks, actions, state_obs, world_obs), kept

class ParseBatch(namedtuple('ParseBatch',
        ['descs', 'actions', 'state_obs', 'world_obs'])):
    @classmethod
    def empty(cls):
        return ParseBatch([], [], [], [])

class SeqBatch(namedtuple('SeqBatch',
        ['desc', 'desc_target', 'init_state_obs', 'last_state_obs',
            'all_state_obs', 'init_world_obs', 'last_world_obs',
            'all_world_obs', 'tasks'])):

    def cuda(self):
        cu_args = [a.cuda() if isinstance(a, Variable) else a for a in self]
        return SeqBatch(*cu_args)

    # TODO dup dup dup
    @classmethod
    def of(cls, batch, dataset):
        tasks, _, state_obs, world_obs = batch

        max_demo_len = max(o.shape[0] for o in state_obs)
        n_state_obs = state_obs[0].shape[1:]
        n_world_obs = world_obs[0].shape[1:]

        desc = []
        init_state_obs = np.zeros((len(tasks),) + n_state_obs)
        last_state_obs = np.zeros((len(tasks),) + n_state_obs)
        all_state_obs = np.zeros((len(tasks), max_demo_len,) + n_state_obs)
        init_world_obs = np.zeros((len(tasks),) + n_world_obs)
        last_world_obs = np.zeros((len(tasks),) + n_world_obs)
        all_world_obs = np.zeros((len(tasks), max_demo_len,) + n_world_obs)
        for task_id in range(len(tasks)):
            demo_len = state_obs[task_id].shape[0]
            desc.append(tasks[task_id].desc)
            init_state_obs[task_id, ...] = state_obs[task_id][0, ...]
            last_state_obs[task_id, ...] = state_obs[task_id][-1, ...]
            all_state_obs[task_id, :demo_len, ...] = state_obs[task_id]
            init_world_obs[task_id, ...] = world_obs[task_id][0, ...]
            last_world_obs[task_id, ...] = world_obs[task_id][-1, ...]
            all_world_obs[task_id, :demo_len, ...] = world_obs[task_id]

        desc_data, desc_target_data = load_desc_data(desc, dataset, target=True)
        init_state_obs_data = torch.FloatTensor(init_state_obs)
        last_state_obs_data = torch.FloatTensor(last_state_obs)
        all_state_obs_data = torch.FloatTensor(all_state_obs)
        init_world_obs_data = torch.FloatTensor(init_world_obs)
        last_world_obs_data = torch.FloatTensor(last_world_obs)
        all_world_obs_data = torch.FloatTensor(all_world_obs)
        out = SeqBatch(
            Variable(desc_data), Variable(desc_target_data),
            Variable(init_state_obs_data), Variable(last_state_obs_data),
            Variable(all_state_obs_data), Variable(init_world_obs_data), 
            Variable(last_world_obs_data), Variable(all_world_obs_data), tasks)

        if FLAGS.gpu:
            out = out.cuda()
        return out

class StepBatch(namedtuple('StepBatch',
        ['init_state_obs', 'state_obs', 'init_world_obs', 'world_obs', 'act',
            'act_pos', 'act_pos_mask', 'final', 'desc_in', 'desc_out_mask',
            'desc_out', 'desc_out_target'])):

    def cuda(self):
        cu_args = [a.cuda() if isinstance(a, Variable) else a for a in self]
        return StepBatch(*cu_args)

    @classmethod
    def of(cls, batch, parse_batch, dataset):
        descs = [task.desc for task in batch.tasks] + parse_batch.descs
        actions = list(batch.actions) + list(parse_batch.actions)
        state_observations = list(batch.state_obs) + list(parse_batch.state_obs)
        world_observations = list(batch.world_obs) + list(parse_batch.world_obs)
        n_tasks = len(descs)

        env_shape = world_observations[0][0].shape[1:]

        init_state_obs = []
        state_obs = []
        init_world_obs = []
        world_obs = []
        act = []
        act_pos = []
        act_pos_mask = []
        final = []
        desc_in = []
        for _ in range(FLAGS.n_batch_steps):
            i_task = np.random.randint(n_tasks)
            i_step = np.random.randint(len(actions[i_task]))
            # TODO yuck
            if isinstance(state_observations[i_task][0], np.ndarray):
                init_state_obs.append(torch.FloatTensor(state_observations[i_task][0]))
                state_obs.append(torch.FloatTensor(state_observations[i_task][i_step]))
                init_world_obs.append(torch.FloatTensor(world_observations[i_task][0]))
                world_obs.append(torch.FloatTensor(world_observations[i_task][i_step]))
                # TODO really yuck
                if FLAGS.gpu:
                    init_state_obs[-1] = init_state_obs[-1].cuda()
                    state_obs[-1] = state_obs[-1].cuda()
                    init_world_obs[-1] = init_world_obs[-1].cuda()
                    world_obs[-1] = world_obs[-1].cuda()
            else:
                init_state_obs.append(state_observations[i_task][0])
                state_obs.append(state_observations[i_task][i_step])
                init_world_obs.append(world_observations[i_task][0])
                world_obs.append(world_observations[i_task][i_step])
            action, action_pos = actions[i_task][i_step]
            act.append(action)
            if action_pos is None:
                act_pos.append(0)
                act_pos_mask.append(0)
            else:
                raveled = int(np.ravel_multi_index(action_pos, env_shape))
                act_pos.append(raveled)
                act_pos_mask.append(1)
            final.append(i_step == len(actions[i_task])-1)
            desc_in.append(descs[i_task])

        init_state_obs_data = torch.stack(init_state_obs)
        state_obs_data = torch.stack(state_obs)
        init_world_obs_data = torch.stack(init_world_obs)
        world_obs_data = torch.stack(world_obs)
        act_data = torch.LongTensor(act)
        act_pos_data = torch.LongTensor(act_pos)
        act_pos_mask_data = torch.FloatTensor(act_pos_mask)
        final_data = torch.FloatTensor(final)
        desc_in_data = load_desc_data(desc_in, dataset)
        out = StepBatch(
            Variable(init_state_obs_data), Variable(state_obs_data), 
            Variable(init_world_obs_data), Variable(world_obs_data),
            Variable(act_data), Variable(act_pos_data),
            Variable(act_pos_mask_data), Variable(final_data),
            Variable(desc_in_data), None, None, None)

        if FLAGS.gpu:
            out = out.cuda()
        return out

class DiskDataset(torch_data.Dataset):
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
            return FLAGS.n_val_batches * self.n_batch
        else:
            return len(self.actions) - FLAGS.n_val_batches * self.n_batch

    def __getitem__(self, i):
        if self.validation:
            i += len(self.actions) - FLAGS.n_val_batches * self.n_batch
        with open(os.path.join(self.cache_dir, 'tasks', 'task%d.pkl' % i), 'rb') as task_f:
            task = pickle.load(task_f)
        state_obs = np.load(os.path.join(self.cache_dir, 'state_obs', 'path%d.npz' % i))['arr_0']
        world_obs = np.load(os.path.join(self.cache_dir, 'world_obs', 'path%d.npz' % i))['arr_0']

        return task, self.actions[i], state_obs, world_obs

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
        task.validate(demo[-1][0])
        actions = [a for s, a, s_ in demo]
        all_obs = [s.obs() for s, a, s_ in demo]
        state_obs, world_obs = zip(*all_obs)
        return task, actions, np.asarray(state_obs), np.asarray(world_obs)

def cache_dataset(env):
    os.mkdir(FLAGS.cache_dir)
    os.mkdir(os.path.join(FLAGS.cache_dir, 'state_obs'))
    os.mkdir(os.path.join(FLAGS.cache_dir, 'world_obs'))
    os.mkdir(os.path.join(FLAGS.cache_dir, 'tasks'))
    actions = []
    for i_task in tqdm(list(range(FLAGS.n_examples))):
        task = env.sample_task()
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

        with open(os.path.join(FLAGS.cache_dir, 'tasks', 'task%d.pkl' % i_task), 'wb') as task_f:
            pickle.dump(task, task_f)

        np.savez(
            os.path.join(FLAGS.cache_dir, 'state_obs', 'path%d.npz' % i_task),
            np.asarray(state_obs))

        np.savez(
            os.path.join(FLAGS.cache_dir, 'world_obs', 'path%d.npz' % i_task),
            np.asarray(world_obs))

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
        hlog.value('setup_data_size', len(data))
        hlog.value('vocab_size', len(vocab))

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
    #assert val_interesting is not None

    return (dataset, val_dataset, val_interesting)

def load_desc_data(descs, dataset, target=False, tokenize=True):
    if tokenize:
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
    tasks, actions, state_obs, world_obs = zip(*items)
    return Batch(tasks, actions, state_obs, world_obs)
