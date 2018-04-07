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
import os
import sys
import torch
import torch.utils.data as torch_data

FLAGS = gflags.FLAGS
gflags.DEFINE_integer('n_setup_examples', 1000, 'num examples to use for building vocab etc')
gflags.DEFINE_integer('n_examples', 2000, 'num training examples to generate')
gflags.DEFINE_integer('n_val_examples', 200, 'num training examples to generate')
gflags.DEFINE_integer('n_batch_examples', 50, 'num full trajectories per batch')
gflags.DEFINE_integer('n_batch_steps', 100, 'num steps per batch')
gflags.DEFINE_integer('n_rollout_max', 50, 'max rollout length')
gflags.DEFINE_integer('n_epochs', 5, 'num training epochs')
gflags.DEFINE_integer('n_flat_passes', 60, 'num passes per epoch for flat policy')
gflags.DEFINE_integer('n_hier_passes', 50, 'num passes per epoch for hierarchical policy')
gflags.DEFINE_integer('n_log', 20, 'num passes after which to log')
gflags.DEFINE_string('cache_dir', '/data/jda/hl3/_cache', 'feature cache directory')
gflags.DEFINE_string('val_cache_dir', '/data/jda/hl3/_val_cache', 'val feature cache directory')
gflags.DEFINE_string('model_dir', '_models', 'model checkpoint directory')
gflags.DEFINE_integer('resume_epoch', None, 'epoch from which to resume')
gflags.DEFINE_boolean('resume_flat', False, 'resume from the flat step')
#gflags.DEFINE_string('cache_dir', None, 'feature cache directory')
gflags.DEFINE_boolean('gpu', True, 'use the gpu')
gflags.DEFINE_boolean('debug', False, 'debug model')

FLAT_TAG = "flat"
HIER_TAG = "hier"

#np.set_printoptions(linewidth=1000, precision=2, suppress=True)

ENV = CraftEnv()
#ENV = NavEnv

def _log(stats):
    count = stats['_count']
    for k, v in stats.items():
        if k == '_count':
            continue
        hlog.value(k, v / count)

def _save(model, index, tag):
    torch.save(
        model.state_dict(), 
        os.path.join(FLAGS.model_dir, '%05d_%s.pth' % (index, tag)))

def _restore(model):
    if FLAGS.resume_flat:
        tag = FLAT_TAG
        first_epoch = FLAGS.resume_epoch
        skip_first_flat = True
    else:
        tag = HIER_TAG
        first_epoch = FLAGS.resume_epoch + 1
        skip_first_flat = False

    model.load_state_dict(torch.load(
        os.path.join(FLAGS.model_dir, '%05d_%s.pth' % (FLAGS.resume_epoch, tag))))

    return first_epoch, skip_first_flat

@profile
def main():
    np.random.seed(0)
    dataset, val_dataset = data.get_dataset(ENV)
    model = Model(ENV, dataset)
    if FLAGS.gpu:
        model.cuda()

    first_epoch = 0
    skip_flat = False
    if FLAGS.resume_epoch is not None:
        first_epoch, skip_flat = _restore(model)

    loader = torch_data.DataLoader(
        dataset, batch_size=FLAGS.n_batch_examples, shuffle=True, 
        num_workers=4,
        collate_fn=lambda items: data.collate(items, dataset))

    vtrain_loader = torch_data.DataLoader(
        dataset, batch_size=FLAGS.n_batch_examples, shuffle=False,
        num_workers=1,
        sampler=list(range(50)),
        collate_fn=lambda items: data.collate(items, dataset))

    val_loader = torch_data.DataLoader(
        val_dataset, batch_size=FLAGS.n_batch_examples, shuffle=False,
        num_workers=1,
        collate_fn=lambda items: data.collate(items, val_dataset))

    loader = vtrain_loader

    @hlog.fn('exec')
    def validate():
        for d, l, n, f in [
                (dataset, vtrain_loader, 'train', lambda m: m.act),
                (dataset, vtrain_loader, 'train_h', lambda m: m.act_hier),
                ]:
                #(val_dataset, val_loader, 'val', lambda m: m.act),
                #(val_dataset, val_loader, 'val_h', lambda m: m.act_hier)
            training.validate(model, d, l, ENV, n, f)

    with hlog.task('train'):
        i_iter = 0
        for i_epoch in hlog.loop('epoch_%03d', range(first_epoch, FLAGS.n_epochs)):
            # flat step
            n_flat_passes = 0 if skip_flat else FLAGS.n_flat_passes
            skip_flat = False
            stats = Counter()
            for i_pass in hlog.loop('flat_%03d', range(n_flat_passes)):
                for i_batch, batch in hlog.loop(
                        'batch_%05d', enumerate(loader), timer=False):
                    seq_batch = data.SeqBatch.of(batch, dataset)
                    step_batch = data.StepBatch.of(batch, dataset)
                    stats += model.train_seq(seq_batch)
                    stats += model.train_step(step_batch)
                    stats['_count'] += 1
                if (i_pass + 1) % FLAGS.n_log == 0:
                    _log(stats)
                    validate()
                    stats = Counter()

            #_save(model, i_epoch, FLAT_TAG)
            #validate()

            # parse
            parses = {}
            with hlog.task('parse'):
                for i_batch, batch in hlog.loop(
                        'batch_%05d', enumerate(loader), timer=False):
                    seq_batch = data.SeqBatch.of(batch, dataset)
                    batch_parses = model.parse(seq_batch)
                    assert not any(k in parses for k in batch_parses)
                    parses.update(batch_parses)

            validate()

            # hier step
            stats = Counter()
            for i_pass in hlog.loop('hier_%03d', range(FLAGS.n_hier_passes)):
                for i_batch, batch in hlog.loop(
                        'batch_%05d', enumerate(loader), timer=False):
                    batch_parses = [parses[task.task_id] for task in batch.tasks]
                    seq_batch = data.StepBatch.of(
                        batch, dataset, hier=True, parses=batch_parses)
                    stats += model.train_hier(seq_batch)
                    stats['_count'] += 1
                if (i_pass + 1) % FLAGS.n_log == 0:
                    _log(stats)
                    validate()
                    stats = Counter()

            _save(model, i_epoch, HIER_TAG)

if __name__ == '__main__':
    gflags.FLAGS(sys.argv)
    main()
