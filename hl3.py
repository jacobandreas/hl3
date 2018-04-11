#!/usr/bin/env python3

from nav.task import NavEnv
from craft.task import CraftEnv
import data
import interact
from misc import fakeprof, hlog, util
from misc.stats import Stats
from model import Model
from trainer import Trainer

from collections import Counter, defaultdict
import gflags
import numpy as np
import os
import sys
import torch
import torch.utils.data as torch_data

FLAGS = gflags.FLAGS
gflags.DEFINE_integer('seed', 0, 'random seed')
gflags.DEFINE_integer('n_examples', 500, 'num training examples to generate')
gflags.DEFINE_integer('n_val_examples', 200, 'num training examples to generate')
gflags.DEFINE_integer('n_batch_examples', 50, 'num full trajectories per batch')
gflags.DEFINE_integer('n_batch_steps', 100, 'num steps per batch')
gflags.DEFINE_integer('n_rollout_max', 50, 'max rollout length')
gflags.DEFINE_integer('n_epochs', 20, 'num training epochs')
gflags.DEFINE_integer('n_flat_passes', 10, 'num passes per epoch for flat policy')
gflags.DEFINE_integer('n_hier_passes', 10, 'num passes per epoch for hierarchical policy')
gflags.DEFINE_integer('n_exec', 10, 'num passes after which to do live rollouts')
gflags.DEFINE_integer('n_parse', 1, 'number of epochs after which to re-parse')
gflags.DEFINE_string('cache_dir', '/data/jda/hl3/_cache', 'feature cache directory')
gflags.DEFINE_string('val_cache_dir', '/data/jda/hl3/_val_cache', 'val feature cache directory')
gflags.DEFINE_string('model_dir', 'models', 'model checkpoint directory')
gflags.DEFINE_string('vis_dir', 'vis', 'visualization')
gflags.DEFINE_integer('resume_epoch', None, 'epoch from which to resume')
gflags.DEFINE_boolean('resume_flat', False, 'resume from the flat step')
gflags.DEFINE_boolean('train_flat_on_parse', False, 'train flat model on subtasks extracted by parser')
#gflags.DEFINE_string('cache_dir', None, 'feature cache directory')
gflags.DEFINE_boolean('gpu', True, 'use the gpu')
gflags.DEFINE_boolean('debug', False, 'debug model')

FLAT_TAG = 'flat'
HIER_TAG = 'hier'

#np.set_printoptions(linewidth=1000, precision=2, suppress=True)

ENV = CraftEnv()
#ENV = NavEnv

def _log(stats):
    for k, v in stats:
        hlog.value(k, v)

def _save(model, trainer, index, tag):
    state_dict = {
        'model': model.state_dict(),
        'trainer': trainer.state_dict()
    }
    torch.save(
        state_dict,
        os.path.join(FLAGS.model_dir, '%05d_%s.pth' % (index, tag)))

def _restore(model, trainer):
    if FLAGS.resume_flat:
        tag = FLAT_TAG
        first_epoch = FLAGS.resume_epoch
        skip_first_flat = True
    else:
        tag = HIER_TAG
        first_epoch = FLAGS.resume_epoch + 1
        skip_first_flat = False

    state_dict = torch.load(
        os.path.join(FLAGS.model_dir, '%05d_%s.pth' % (FLAGS.resume_epoch, tag)))

    model.load_state_dict(state_dict['model'])
    trainer.load_state_dict(state_dict['trainer'])

    return first_epoch, skip_first_flat

@profile
def main():
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    dataset, val_dataset = data.get_dataset(ENV)
    model = Model(ENV, dataset)
    if FLAGS.gpu:
        model.cuda()
    trainer = Trainer(model)

    first_epoch = 0
    skip_flat = False
    if FLAGS.resume_epoch is not None:
        first_epoch, skip_flat = _restore(model, trainer)

    loader = torch_data.DataLoader(
        dataset, batch_size=FLAGS.n_batch_examples, shuffle=True, 
        num_workers=2,
        collate_fn=lambda items: data.collate(items, dataset))

    vtrain_loader = torch_data.DataLoader(
        dataset, batch_size=FLAGS.n_batch_examples, shuffle=False,
        num_workers=1,
        sampler=list(range(FLAGS.n_val_examples)),
        collate_fn=lambda items: data.collate(items, dataset))

    val_loader = torch_data.DataLoader(
        val_dataset, batch_size=FLAGS.n_batch_examples, shuffle=False,
        num_workers=1,
        collate_fn=lambda items: data.collate(items, val_dataset))

    @hlog.fn('exec')
    def execute():
        for d, l, n, f in [
                (dataset, vtrain_loader, 'train', lambda m: m.act),
                #(dataset, vtrain_loader, 'train_h', lambda m: m.act_hier),
                (val_dataset, val_loader, 'val', lambda m: m.act),
                #(val_dataset, val_loader, 'val_h', lambda m: m.act_hier),
                ]:
            interact.execute(model, d, l, ENV, n, f, dump=True)

    @hlog.fn('train', timer=False)
    def train_step(batch):
        if FLAGS.train_flat_on_parse:
            batch_parses = [parses[task.task_id] for task in batch.tasks]
        else:
            batch_parses = None
        seq_batch = data.SeqBatch.of(batch, dataset, parses=batch_parses)
        step_batch = data.StepBatch.of(batch, dataset, parses=batch_parses)
        step_loss, step_stats = model.score_step(step_batch)
        seq_loss, seq_stats = model.score_seq(seq_batch)
        stats = Stats() + step_stats + seq_stats
        loss = step_loss + seq_loss
        trainer.step(train_loss=loss)
        return stats

    @hlog.fn('val', timer=False)
    def val_step():
        stats = Stats()
        loss = 0
        for i_batch, batch in enumerate(val_loader):
            seq_batch = data.SeqBatch.of(batch, dataset)
            step_batch = data.StepBatch.of(batch, dataset)
            step_loss, step_stats = model.score_step(step_batch)
            seq_loss, seq_stats = model.score_seq(seq_batch)
            stats += Stats() + step_stats + seq_stats
            loss += seq_loss + step_loss
        trainer.step(val_loss=loss)
        _log(stats)

    execute()

    with hlog.task('learn'):
        i_iter = 0
        parses = defaultdict(list)
        for i_epoch in hlog.loop('epoch_%03d', range(first_epoch, FLAGS.n_epochs)):

            # flat step
            n_flat_passes = 0 if skip_flat else FLAGS.n_flat_passes
            skip_flat = False
            for i_pass in hlog.loop('flat_%03d', range(n_flat_passes)):
                stats = Stats()
                for i_batch, batch in hlog.loop(
                        'batch_%05d', enumerate(loader), timer=False):
                    stats += train_step(batch)
                _log(stats)

                val_step()
                if (i_pass + 1) % FLAGS.n_exec == 0:
                    execute()

            _save(model, trainer, i_epoch, FLAT_TAG)

            # parse
            if (i_epoch + 1) % FLAGS.n_parse == 0:
                parses = {}
                with hlog.task('parse'):
                    for i_batch, batch in hlog.loop(
                            'batch_%05d', enumerate(loader), timer=False):
                        seq_batch = data.SeqBatch.of(batch, dataset)
                        batch_parses = model.parse(seq_batch)
                        assert not any(k in parses for k in batch_parses)
                        parses.update(batch_parses)

            ### # hier step
            ### stats = Counter()
            ### for i_pass in hlog.loop('hier_%03d', range(FLAGS.n_hier_passes)):
            ###     for i_batch, batch in hlog.loop(
            ###             'batch_%05d', enumerate(loader), timer=False):
            ###         batch_parses = [parses[task.task_id] for task in batch.tasks]
            ###         seq_batch = data.StepBatch.of(
            ###             batch, dataset, hier=True, parses=batch_parses)
            ###         stats += model.train_hier(seq_batch)
            ###         stats['_count'] += 1
            ###     if (i_pass + 1) % FLAGS.n_log == 0:
            ###         _log(stats)
            ###         validate()
            ###         stats = Counter()

            _save(model, trainer, i_epoch, HIER_TAG)

if __name__ == '__main__':
    gflags.FLAGS(sys.argv)
    main()
