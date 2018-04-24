from misc.util import unwrap

import itertools as it
from torch import nn, optim
from torch.optim import lr_scheduler as opt_sched

class Trainer(object):
    def __init__(self, model):
        self._opt = optim.Adam(
            it.chain(
                model._featurizer.parameters(), model._flat_policy.parameters(),
                model._segmenter.parameters(), model._describer.parameters()),
            5e-3)
        self._sched = opt_sched.ReduceLROnPlateau(self._opt, factor=0.5, verbose=True)

        self._hier_opt = optim.Adam(model._hier_policy.parameters())
        #self._hier_sched = opt_sched.ReduceLROnPlateau(self._hier_opt)

    def step(self, train_loss=None, val_loss=None, hier_loss=None):
        if train_loss is not None:
            self._opt.zero_grad()
            train_loss.backward()
            self._opt.step()

        if val_loss is not None:
            self._sched.step(unwrap(val_loss)[0])

        if hier_loss is not None:
            self._hier_opt.zero_grad()
            hier_loss.backward()
            self._hier_opt.step()

    def state_dict(self):
        return self._opt.state_dict()

    def load_state_dict(self, state_dict):
        self._opt.load_state_dict(state_dict)
