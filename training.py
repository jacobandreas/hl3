import data
import ling
from misc import hlog

from collections import Counter
from itertools import islice
import numpy as np
import torch

@profile
def rollout(task, model, dataset, env, config):
    desc = ling.tokenize(task.desc, dataset.vocab)
    desc_data = torch.zeros(len(desc), 1, len(dataset.vocab))
    for i, tok in enumerate(desc):
        desc_data[i, 0, tok] = 1
    desc_var = torch.autograd.Variable(desc_data.cuda())

    state = task.init_state
    steps = []
    for _ in range(config.N_ROLLOUT_MAX):
        obs_data = [state.features()]
        obs_data = torch.FloatTensor(obs_data)
        batch = data.PolicyBatch(
            desc_var,
            None,
            torch.autograd.Variable(obs_data.cuda()),
            None, None)
        action, = model.act(batch)
        s_ = state.step(action)
        steps.append((state, action, s_))
        state = s_
        if action == env.STOP:
            break
    return steps

def validate(model, dataset, parse_ex, env, config):
    val_loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.N_BATCH_EXAMPLES, shuffle=True,
        num_workers=2,
        collate_fn=lambda items: data.collate(items, dataset, config))

    val_stats = Counter()
    val_count = 0
    for pol_batch, seg_batch in islice(val_loader, 100):
        pol_batch = pol_batch.cuda()
        seg_batch = seg_batch.cuda()
        val_stats += model.evaluate(pol_batch, seg_batch, train=False)
        val_count += 1
    for k, v in val_stats.items():
        hlog.value(k, v / val_count)

    #with hlog.task('rollout'):
    #    eval_task = env.sample_task()
    #    steps = rollout(eval_task, model, dataset, env, config)
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

    with hlog.task('parse'):
        _, seg_batch = data.collate([parse_ex], dataset, config)
        parse = model.parse(seg_batch.cuda())
        hlog.value('result', parse)
        #split = parse[2][0][1]
        split = parse[0][0][1]
        hlog.value('actions', (
            ' '.join(env.action_name(a) for a in parse_ex[1][:split]), 
            ' '.join(env.action_name(a) for a in parse_ex[1][split:])))
