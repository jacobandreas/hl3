from collections import namedtuple
import numpy as np

class NavState(namedtuple('NavState', ['init_pos', 'pos'])):
    def features(self):
        init_feats = np.zeros((NavEnv.SIZE, NavEnv.SIZE))
        init_feats[self.init_pos] = 1
        feats = np.zeros((NavEnv.SIZE, NavEnv.SIZE))
        feats[self.pos] = 1
        return np.concatenate((init_feats.ravel(), feats.ravel()))

    def step(self, action):
        r, c = self.pos
        if action == NavEnv.EAST:
            assert c < NavEnv.SIZE - 1
            c += 1
        elif action == NavEnv.SOUTH:
            assert r < NavEnv.SIZE - 1
            r += 1
        return NavState(self.init_pos, (r, c))

    def with_init(self, state):
        return NavState(state.pos, self.pos)

class NavTask(namedtuple('NavTask', ['desc', 'init_state', 'actions'])):
    def demonstration(self):
        out = []
        state = self.init_state
        #print()
        for action in self.actions:
            #print(action)
            new_state = state.step(action)
            out.append((state, action, new_state))
            state = new_state
        out.append((state, NavEnv.STOP, None))
        return out

class NavEnv(object):
    SIZE = 3

    EAST = 0
    SOUTH = 1
    STOP = 2
    _action_names = {
        EAST: 'e',
        SOUTH: 's',
        STOP: '.',
    }

    TASKS = [
        NavTask('east', NavState((0, 0), (0, 0)), [EAST] * (SIZE-1)),
        NavTask('east', NavState((SIZE-1, 0), (SIZE-1, 0)), [EAST] * (SIZE-1)),
        NavTask('south', NavState((0, 0), (0, 0)), [SOUTH] * (SIZE-1)),
        NavTask('south', NavState((0, SIZE-1), (0, SIZE-1)), [SOUTH] * (SIZE-1)),
        NavTask('southeast', NavState((0, 0), (0, 0)), [EAST] * (SIZE-1) + [SOUTH] * (SIZE-1)),
        NavTask('southeast', NavState((0, 0), (0, 0)), [SOUTH] * (SIZE-1) + [EAST] * (SIZE-1))
    ]

    n_features = 2 * SIZE * SIZE
    n_actions = 3

    @classmethod
    def sample_task(cls):
        return cls.TASKS[np.random.randint(len(cls.TASKS))]

    @classmethod
    def action_name(cls, action):
        return cls._action_names[action]
