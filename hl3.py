#!/usr/bin/env python3

from nav.task import NavEnv
from craft.task import CraftEnv
import data
from misc import fakeprof, hlog, util
from model import Model
import training

from collections import Counter
import numpy as np
import torch.utils.data as torch_data

config = util.Struct()
###config.N_SETUP_EXAMPLES = 1000
###config.N_EXAMPLES = 20000
config.N_SETUP_EXAMPLES = 100
config.N_EXAMPLES = 2000
config.N_BATCH_EXAMPLES = 10
config.N_BATCH_STEPS = 50
config.N_ROLLOUT_MAX = 1000
config.N_EPOCHS = 20
config.N_LOG = 20
config.CACHE_DIR = '_cache'
###config.CACHE_DIR = '/data/jda/hl3/_debug_cache'
#config.CACHE_DIR = None
config.GPU = False

ENV = CraftEnv
#ENV = NavEnv

def _log(stats, model):
    for k, v in stats.items():
        hlog.value(k, v / config.N_LOG)
    #with hlog.task('val'):
    #    training.validate(model, val_dataset, parse_ex, ENV, config)

@profile
def main():
    np.random.seed(0)
    dataset, val_dataset, parse_ex = data.get_dataset(ENV, config)
    model = Model(ENV, dataset, config)

    loader = torch_data.DataLoader(
        dataset, batch_size=config.N_BATCH_EXAMPLES, shuffle=True, 
        num_workers=4,
        collate_fn=lambda items: data.collate(items, dataset, config))

    with hlog.task('train'):
        stats = Counter()
        i_iter = 0
        for i_epoch in hlog.loop('epoch_%05d', range(config.N_EPOCHS)):
            for i_batch, batch in hlog.loop('batch_%05d', enumerate(loader)):

                with hlog.task('e-step', timer=False):
                    seq_batch = data.SeqBatch.of(batch, dataset, config)
                    #parses = model.parse(seq_batch)
                    parses = None
                    step_batch = data.StepBatch.of(batch, parses, dataset, config)

                with hlog.task('m-step', timer=False):
                    stats += model.train_step(step_batch)
                    stats += model.train_seq(seq_batch)

                i_iter += 1
                if i_iter % config.N_LOG == 0:
                    _log(stats, model)
                    #model.parse(seq_batch)
                    stats = Counter()

                    seq_batch = data.SeqBatch.of([[p] for p in parse_ex], val_dataset, config)
                    model.parse(seq_batch)

if __name__ == '__main__':
    main()
