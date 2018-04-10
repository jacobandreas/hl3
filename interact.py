import data
import ling
from misc import hlog

from collections import Counter
import gflags
from itertools import islice
import numpy as np
import torch
from torch.autograd import Variable

FLAGS = gflags.FLAGS

@profile
def rollout(tasks, model, dataset, env, act_fn):
    descs = [task.desc for task in tasks]
    states = [task.init_state for task in tasks]
    init_obs = [state.obs() for state in states]
    steps = [[] for _ in tasks]
    done = [False for _ in tasks]
    for _ in range(FLAGS.n_rollout_max):
        obs = [state.obs() for state in states]
        batch = data.StepBatch.for_states(init_obs, obs, descs, dataset)
        actions = act_fn(model)(batch, sample=False)

        states_ = [None for _ in states]
        for i in range(len(tasks)):
            s = states[i]
            if done[i]:
                states_[i] = s
                continue
            a = actions[i]
            if a[0] != env.GO:
                a = (a[0], None)
            s_ = s.step(a)
            steps[i].append((s, a, s_))
            states_[i] = s_
            if a[0] == env.STOP:
                done[i] = True
                states_[i] = s
        states = states_
        if all(done):
            break
    return steps

def visualize(scenes, prefix):
    for name, (scene, label) in scenes.items():
        with open('%s_%s.json' % (prefix, name), 'w') as scene_f:
            scene.dump(label, scene_f)

def execute(model, dataset, loader, env, log_name, act_fn, dump=False):
    score = 0.
    tot = 0.
    for i_batch, batch in enumerate(loader):
        #task_groups = [batch.tasks[i:i+5] for i in range(0, len(batch.tasks), 5)]
        task_groups = [batch.tasks]
        steps = [rollout(tg, model, dataset, env, act_fn) for tg in task_groups]
        steps = sum(steps, [])
        last_states = [ss[-1][0] for ss in steps]
        last_actions = [ss[-1][1][0] for ss in steps]
        scores = [t.validate(s) for t, s in zip(batch.tasks, last_states)]
        scores = [0. if a != env.STOP else s for s, a in zip(scores, last_actions)]
        score += sum(scores)
        tot += len(scores)
        if dump and i_batch == 0:
            # TODO magic
            for i_task in range(5):
                task = batch.tasks[i_task]
                with hlog.task(str(i_task), timer=False):
                    hlog.value('desc', dataset.render_desc(task.desc))
                    hlog.value('gold', [a for s, a, s_ in task.demonstration()])
                    hlog.value('pred', [a for s, a, s_ in steps[i_task]])
                    hlog.value('score_here', scores[i_task])
                    visualize(
                        {
                            'before': (task.scene_before, task.desc),
                            'after': (last_states[i_task].to_scene(), task.desc),
                        },
                        'vis/scenes/%s_%d' % (log_name, i_task))

    score /= tot
    with hlog.task(log_name, timer=False):
        hlog.value('score', score)
