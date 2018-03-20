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
def rollout(task, model, dataset, env):
    desc = ling.tokenize(task.desc, dataset.vocab)
    desc_data = torch.zeros(len(desc), 1, len(dataset.vocab))
    for i, tok in enumerate(desc):
        desc_data[i, 0, tok] = 1
    desc = torch.autograd.Variable(desc_data.cuda())

    state = task.init_state
    init_features = Variable(torch.FloatTensor([state.features()]))
    steps = []
    for _ in range(FLAGS.n_rollout_max):
        features = Variable(torch.FloatTensor([state.features()]))
        batch = data.StepBatch(
            init_features,
            features,
            None, None,
            desc,
            None, None, None)
        batch = batch.cuda() 
        action, = model.act(batch, sample=False)
        s_ = state.step(action)
        steps.append((state, action, s_))
        state = s_
        if action == env.STOP:
            break
    return steps

def validate(model, dataset, env):
    val_loader = torch.utils.data.DataLoader(
        dataset, batch_size=FLAGS.n_batch_examples, shuffle=True,
        num_workers=2,
        collate_fn=lambda items: data.collate(items, dataset))

    score = 0.
    tot = 0.
    for batch in islice(val_loader, 10):
        for task in batch.tasks:
            print(task.desc)
            print('gold', [a for s, a, s_ in task.demonstration()])
            steps = rollout(task, model, dataset, env)
            print('pred', [a for s, a, s_ in steps])
            last_state = steps[-1][0]
            score_here = task.validate(last_state)
            print(score_here)
            print()
            score += score_here
            tot += 1
    score /= tot
    hlog.value('score', score)
    return {'score': score}

    #val_stats = Counter()
    #val_count = 0
    #for pol_batch, seg_batch in islice(val_loader, 100):
    #    pol_batch = pol_batch.cuda()
    #    seg_batch = seg_batch.cuda()
    #    val_stats += model.evaluate(pol_batch, seg_batch, train=False)
    #    val_count += 1
    #for k, v in val_stats.items():
    #    hlog.value(k, v / val_count)

    #with hlog.task('rollout'):
    #    eval_task = env.sample_task()
    #    steps = rollout(eval_task, model, dataset, env)
    #    actions = [s[1] for s in steps]
    #    last_state = steps[-1][0]
    #    last_scene = last_state.to_scene()
    #    hlog.value('desc', eval_task.desc)
    #    hlog.value('pred', ' '.join(env.action_name(a) for a in actions))
    #    hlog.value('gold', ' '.join(env.action_name(a) for s, a, s_ in eval_task.demonstration()))
    #    with open('vis/before.json', 'w') as scene_f:
    #        eval_task.scene_before.dump(scene_f)
    #    with open('vis/after.json', 'w') as scene_f:
    #        last_scene.dump(scene_f)
    #    with open('vis/after_gold.json', 'w') as scene_f:
    #        eval_task.scene_after.dump(scene_f)
