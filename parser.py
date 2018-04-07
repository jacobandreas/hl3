from data import StepBatch
from misc.util import flatten, unwrap

import gflags
import numpy as np
import torch

FLAGS = gflags.FLAGS
gflags.DEFINE_integer('n_parse_splits', 2, 'number of parser splits to consider')
gflags.DEFINE_integer('n_parse_descs', 2, 'number of parser span descriptions to consider')
gflags.DEFINE_integer('parse_desc_max_len', 20, 'max length of parser span description')

class TopDownParser(object):
    def __init__(self, model, env, dataset):
        self._model = model
        self._env = env
        self._dataset = dataset

    def propose_splits(self, seq_batch, i_task, start, end):
        size = len(seq_batch.act[i_task])
        mid_state_obs = seq_batch.obs[0][i_task, start+1:end, :]
        init_state_obs = seq_batch.obs[0][i_task, start, :].expand_as(mid_state_obs)
        last_state_obs = seq_batch.obs[0][i_task, end, :].expand_as(mid_state_obs)

        feats_before = self._model._featurizer(
            (init_state_obs, None), (mid_state_obs, None), skip_world=True)
        feats_after = self._model._featurizer(
            (mid_state_obs, None), (last_state_obs, None), skip_world=True)
        scores = (
            self._model._segmenter(feats_before)
            + self._model._segmenter(feats_after))
        _, top = scores.topk(min(FLAGS.n_parse_splits, scores.shape[0]))
        return start + 1 + top

    def propose_descs(self, seq_batch, indices):
        n = FLAGS.n_parse_descs
        max_len = FLAGS.parse_desc_max_len
        init_state_obs = torch.cat(
            [seq_batch.obs[0][b, i, :].expand(n, -1) for b, i, j in indices])
        state_obs = torch.cat(
            [seq_batch.obs[0][b, j, :].expand(n, -1) for b, i, j in indices])
        state_feats, _ = self._model._featurizer(
            (init_state_obs, None), (state_obs, None), skip_world=True)
        descs = self._model._describer.decode(
            state_feats.unsqueeze(0), max_len, sample=True)
        descs = [descs[i:i+n] for i in range(0, len(descs), n)]
        return descs

    def score_span(self, seq_batch, i_task, start, end, desc):
        step_batch = StepBatch.for_seq(
            seq_batch, i_task, start, end+1, self._dataset, desc)

        # patch in STOP action
        step_batch.act[0][-1] = self._env.STOP
        step_batch.act[1][-1] = 0
        step_batch.act_mask[0][-1] = 1
        step_batch.act_mask[1][-1] = 0

        feats = self._model._featurizer(step_batch.init_obs, step_batch.obs)
        return self._model._logprob_of(self._model._flat_policy, feats, step_batch)

    def parse(self, seq_batch):
        out = {}
        for i_task, task in enumerate(seq_batch.tasks):
            actions = seq_batch.act[i_task]
            assert task.task_id not in out
            out[task.task_id] = self._parse_inner(
                seq_batch, i_task, 0, len(actions) - 1, 1, task.desc)
        return out

    def best_desc(self, seq_batch, i_task, start, end, descs):
        scores = [
            self.score_span(seq_batch, i_task, start, end, desc).sum()
            for desc in descs]
        scores = [unwrap(score)[0] for score in scores]
        return min(zip(scores, descs))

    # TODO cleanup
    @profile
    def _parse_inner(self, seq_batch, i_task, start, end, remaining_depth, top_desc):
        if remaining_depth <= 0:
            return []
        if end - start < 2:
            return []

        task = seq_batch.tasks[i_task]

        # TODO only this segment
        root_scores = self.score_span(
            seq_batch, i_task, 0, len(seq_batch.act[i_task])-1, top_desc)

        splits = unwrap(self.propose_splits(seq_batch, i_task, start, end))
        splits = [int(k) for k in splits]

        indices = [[(i_task, start, k), (i_task, k, end)] for k in splits]
        indices = sum(indices, [])
        descs = self.propose_descs(seq_batch, indices)
        desc_pairs = [(descs[2*i], descs[2*i+1]) for i in range(len(splits))]

        candidates = []
        for k, (descs1, descs2) in zip(splits, desc_pairs):

            s1c, desc1 = self.best_desc(seq_batch, i_task, start, k, descs1)
            s2c, desc2 = self.best_desc(seq_batch, i_task, k, end, descs2)

            s1p, = unwrap(root_scores[start:k].sum())
            s2p, = unwrap(root_scores[k:end].sum())

            pick = [None, None]
            if s1c < s1p:
                s1 = s1c
                pick[0] = desc1
            else:
                s1 = s1p

            if s2c < s2p:
                s2 = s2c
                pick[1] = desc2
            else:
                s2 = s2p

            candidates.append((s1+s2, k, tuple(pick)))

        if len(candidates) == 0:
            return []

        score, split, (d1, d2) = min(candidates)
        actions = [a for a, ap in seq_batch.act[i_task]]

        out = [
            (d1, (start, split)),
            (d2, (split, end))
        ]
        #if not (d1 is None and d2 is None): # and np.random.random() < 0.05:
        #    print(
        #        self._dataset.render_desc(top_desc),
        #        ':', 
        #        self._dataset.render_desc(d1) if d1 else '_', 
        #        '>', 
        #        self._dataset.render_desc(d2) if d2 else '_')
        #    print(actions[start:split], actions[split:end])
        #    print()

        for start_, end_, desc_ in [(start, split, d1), (split, end, d2)]:
            if desc_ is None:
                desc_ = top_desc
            out += self._parse_inner(
                seq_batch, i_task, start_, end_, remaining_depth-1, desc_)

        return out
