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
gflags.DEFINE_integer('n_setup_examples', 1000, 'num examples to use for building vocab etc')
gflags.DEFINE_integer('n_examples', 2000, 'num training examples to generate')
gflags.DEFINE_integer('n_batch_examples', 50, 'num full trajectories per batch')
gflags.DEFINE_integer('n_batch_steps', 100, 'num steps per batch')
gflags.DEFINE_integer('n_rollout_max', 50, 'max rollout length')
gflags.DEFINE_integer('n_val_batches', 1, 'num validation batches')
gflags.DEFINE_integer('n_epochs', 1, 'num training epochs')
gflags.DEFINE_integer('n_flat_passes', 20, 'num passes per epoch for flat policy')
gflags.DEFINE_integer('n_hier_passes', 1, 'num passes per epoch for hierarchical policy')
#gflags.DEFINE_integer('n_log', 100, 'num passes after which to log')
gflags.DEFINE_string('cache_dir', '/data/jda/hl3/_cache', 'feature cache directory')
gflags.DEFINE_string('model_dir', '_models', 'model checkpoint directory')
gflags.DEFINE_integer('resume_epoch', None, 'epoch from which to resume')
gflags.DEFINE_boolean('resume_flat', False, 'resume from the flat step')
#gflags.DEFINE_string('cache_dir', None, 'feature cache directory')
gflags.DEFINE_boolean('gpu', True, 'use the gpu')
gflags.DEFINE_boolean('debug', False, 'debug model')

#np.set_printoptions(linewidth=1000, precision=2, suppress=True)

ENV = CraftEnv
#ENV = NavEnv

def _log(stats, model):
    for k, v in stats.items():
        hlog.value(k, v / FLAGS.n_log)

@profile
def main():
    np.random.seed(0)
    dataset, val_dataset, parse_ex = data.get_dataset(ENV)
    model = Model(ENV, dataset)

    loader = torch_data.DataLoader(
        dataset, batch_size=FLAGS.n_batch_examples, shuffle=True, 
        num_workers=4,
        collate_fn=lambda items: data.collate(items, dataset))

    val_loader = torch_data.DataLoader(
        val_dataset, batch_size=FLAGS.n_batch_examples, shuffle=True,
        num_workers=1,
        collate_fn=lambda items: data.collate(items, val_dataset))

    with hlog.task('train'):
        i_iter = 0
        for i_epoch in hlog.loop('epoch_%03d', range(FLAGS.n_epochs)):
            # flat step
            for i_pass in hlog.loop('flat_%03d', range(FLAGS.n_flat_passes)):
                stats = Counter()
                for i_batch, batch in hlog.loop(
                        'batch_%05d', enumerate(loader), timer=False):
                    seq_batch = data.SeqBatch.of(batch)
                    step_batch = data.StepBatch.of(batch, parses=None, dataset)
                    stats += model.train_seq(seq_batch)
                    stats += model.train_step(step_batch)
                _log(stats, model)

            # parse
            pass

            # hier step
            for i_pass in hlog.loop('hier_%03d', range(FLAGS.n_hier_passes)):
                pass

if __name__ == '__main__':
    gflags.FLAGS(sys.argv)
    main()
