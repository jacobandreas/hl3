from collections import defaultdict
import copy
import numpy as np

class Stats(object):
    def __init__(self):
        self._contents = {}

    def __add__(self, other):
        out = Stats()
        out._contents = copy.deepcopy(self._contents)
        if isinstance(other, Stats):
            for k in other._contents:
                if k in out._contents:
                    out._contents[k] += other._contents[k]
                else:
                    out._contents[k] = other._contents[k]
        elif isinstance(other, dict):
            for k, v in other.items():
                if k not in out._contents:
                    out._contents[k] = [v]
                else:
                    out._contents[k].append(v)
        else:
            assert False
        return out

    def __iter__(self):
        for k in sorted(self._contents.keys()):
            vs = self._contents[k]
            yield k, '%0.4f ± %0.2f' % (np.mean(vs), np.std(vs))
