#!/usr/bin/env python3

from nav.task import NavEnv
from craft.task import CraftEnv
import data
from misc import fakeprof, hlog, util
from model import Model
import training

from collections import Counter
import gflags
import numpy as np
import sys
import torch.utils.data as torch_data

FLAGS = gflags.FLAGS
gflags.DEFINE_integer('n_setup_examples', 1000, 'number of examples to use for building vocab etc')
gflags.DEFINE_integer('n_examples', 20000, 'number of training examples to generate')
gflags.DEFINE_integer('n_batch_examples', 10, 'number of full trajectories per batch')
gflags.DEFINE_integer('n_batch_steps', 50, 'number of steps per batch')
gflags.DEFINE_integer('n_rollout_max', 1000, 'max rollout length')
gflags.DEFINE_integer('n_epochs', 20, 'number of training epochs')
gflags.DEFINE_integer('n_log', 100, 'logging frequency')
gflags.DEFINE_string('cache_dir', '/data/jda/hl3/_cache', 'feature cache directory')
gflags.DEFINE_boolean('gpu', True, 'use the gpu')

ENV = CraftEnv
#ENV = NavEnv

def _log(stats, model):
    for k, v in stats.items():
        hlog.value(k, v / FLAGS.n_log)
    #with hlog.task('val'):
    #    training.validate(model, val_dataset, parse_ex, ENV)

@profile
def main():
    np.random.seed(0)
    dataset, val_dataset, parse_ex = data.get_dataset(ENV)
    model = Model(ENV, dataset)

    loader = torch_data.DataLoader(
        dataset, batch_size=FLAGS.n_batch_examples, shuffle=True, 
        num_workers=4,
        collate_fn=lambda items: data.collate(items, dataset))

    with hlog.task('train'):
        stats = Counter()
        i_iter = 0
        for i_epoch in hlog.loop('epoch_%05d', range(FLAGS.n_epochs)):
            for i_batch, batch in hlog.loop('batch_%05d', enumerate(loader), timer=False):

                with hlog.task('e-step', timer=False):
                    seq_batch = data.SeqBatch.of(batch, dataset)
                    #parses = model.parse(seq_batch)
                    parses = None
                    step_batch = data.StepBatch.of(batch, parses, dataset)

                with hlog.task('m-step', timer=False):
                    stats += model.train_step(step_batch)
                    stats += model.train_seq(seq_batch)

                i_iter += 1
                if i_iter % FLAGS.n_log == 0:
                    _log(stats, model)
                    #model.parse(seq_batch)
                    stats = Counter()

                    seq_batch = data.SeqBatch.of([[p] for p in parse_ex], val_dataset)
                    model.parse(seq_batch)

if __name__ == '__main__':
    gflags.FLAGS(sys.argv)
    main()
