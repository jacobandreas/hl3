#!/usr/bin/env python3

from craft.builder import BlockType, Scene

from collections import Counter, namedtuple
import numpy as np
from skimage.measure import block_reduce

def dist(pos1, pos2):
    assert len(pos1) == len(pos2)
    return sum(abs(q1 - q2) for q1, q2 in zip(pos1, pos2))

class CraftEnv(object):
    n_actions = 9

    @classmethod
    def sample_task(self):
        while True:
            try:
                task = Task.sample()
                break
            except AssertionError as e:
                pass
        return task

class CraftState(namedtuple('CraftState', ['blocks', 'pos', 'mat'])):
    WEST = 0
    EAST = 1
    DOWN = 2
    UP = 3
    SOUTH = 4
    NORTH = 5
    ADD = 6
    REMOVE = 7
    CLONE = 8

    @classmethod
    def from_scene(cls, scene, pos):
        blocks = np.zeros(scene.size + (len(BlockType.enumerate()),))
        for block in scene.blocks():
            blocks[block.pos + (block.block_type.mat_id(),)] = 1
        return CraftState(blocks, pos, 0)

    def move(self, delta, act):
        npos = tuple(p + d for p, d in zip(self.pos, delta))
        ok = all(n >= 0 for n in npos) and (n < s for n, s in zip(npos, self.blocks.shape))
        if not ok:
            return self, act
        return self._replace(pos=npos), act

    def west(self):
        return self.move((-1, 0, 0), self.WEST)

    def east(self):
        return self.move((1, 0, 0), self.EAST)

    def down(self):
        return self.move((0, -1, 0), self.DOWN)

    def up(self):
        return self.move((0, 1, 0), self.UP)

    def south(self):
        return self.move((0, 0, -1), self.SOUTH)

    def north(self):
        return self.move((0, 0, 1), self.NORTH)

    def add(self):
        if self.blocks[self.pos].any():
            return self, self.ADD
        blocks = self.blocks.copy()
        blocks[self.pos + (self.mat,)] = 1
        return self._replace(blocks=blocks), self.ADD

    def remove(self):
        if not self.blocks[self.pos].any():
            return self, self.REMOVE
        blocks = self.blocks.copy()
        blocks[self.pos] = 0
        return self._replace(blocks=blocks), self.REMOVE

    def features(self):
        from math import ceil, floor

        def pad_slice(w, h, d, f, x, y, z, data):
            assert data.shape == (w, h, d, f)
            e = np.zeros((w+4, h+4, d+4, f))
            e[2:-2, 2:-2, 2:-2, :] = data
            ft = e[x-2+2:x+3+2, y-2+2:y+3+2, z-2+2:z+3+2, :]
            assert ft.shape[:3] == (5, 5, 5)
            return ft

        x, y, z = self.pos

        p0 = self.blocks
        w, h, d, f = p0.shape
        f0 = pad_slice(w, h, d, f, x, y, z, p0)

        w1, h1, d1 = ceil(w/3), ceil(h/3), ceil(d/3)
        x1, y1, z1 = floor(x/3), floor(y/3), floor(z/3)
        p1 = block_reduce(p0, (3, 3, 3, 1), func=np.mean)
        f1 = pad_slice(w1, h1, d1, f, x1, y1, z1, p1)

        w2, h2, d2 = ceil(w1/3), ceil(h1/3), ceil(d1/3)
        x2, y2, z2 = floor(x1/3), floor(y1/3), floor(z1/3)
        p2 = block_reduce(p1, (3, 3, 3, 1), func=np.mean)
        f2 = pad_slice(w2, h2, d2, f, x2, y2, z2, p2)

        return np.concatenate((f0.ravel(), f1.ravel(), f2.ravel()))

class Task(object):
    FIND = 0
    ADD = 1
    REMOVE = 2

    _actions = [FIND, ADD, REMOVE]
    _descriptions = {
        FIND: 'find',
        ADD: 'add',
        REMOVE: 'remove'
    }

    @classmethod
    def sample(cls):
        scene1 = Scene.sample()
        parts = list(scene1.parts())
        part = parts[np.random.randint(len(parts))]
        scene2 = scene1.remove(part)
        action = cls._actions[np.random.randint(len(cls._actions))]
        descs = list(set(part.descriptions(top=True)))
        desc = descs[np.random.randint(len(descs))]

        if action == cls.FIND or action == cls.REMOVE:
            scene_before = scene1
            scene_after = scene2
        else:
            assert action == cls.ADD
            scene_before = scene2
            scene_after = scene1

        return Task(action, part, desc, scene_before, scene_after)

    def __init__(self, action, part, desc, scene_before, scene_after):
        self.action = action
        self.part = part
        self._desc = desc
        self.desc = next(self.descriptions())
        self.scene_before = scene_before
        self.scene_after = scene_after

    def descriptions(self):
        yield '%s %s' % (self._descriptions[self.action], self._desc)

    def dump(self):
        with open('../vis/scene.json', 'w') as scene_f:
            self.scene_before.dump(scene_f)

    def demonstration(self):
        init_pos = [np.random.randint(dim) for dim in self.scene_before.size]
        init_state = CraftState.from_scene(self.scene_before, tuple(init_pos))
        if self.action == self.FIND:
            demo = self.demonstrate_find(init_state)
        else:
            demo = self.demonstrate_change(init_state)
        return demo

    def demonstrate_find(self, state):
        return self.go_to(state, self.part.pos)

    def demonstrate_change(self, state):
        blocks_before = set(self.scene_before.blocks())
        blocks_after = set(self.scene_after.blocks())
        to_add = blocks_after - blocks_before
        to_remove = blocks_before - blocks_after
        if len(to_remove) == 0:
            remaining = to_add
            add = True
        elif len(to_add) == 0:
            remaining = to_remove
            add = False
        else:
            assert False

        demo = []
        while len(remaining) > 0:
            nearest = min(remaining, key=lambda x: dist(x.pos, state.pos))
            remaining.remove(nearest)
            demo += self.go_to(state, nearest.pos)
            if add:
                s, a = state.add()
            else:
                s, a = state.remove()
            demo.append((state, a, s))
            state = s
        return demo

    def go_to(self, state, dest):
        demo = []
        while state.pos != dest:
            if dest[0] < state.pos[0]:
                s, a = state.west()
            elif dest[0] > state.pos[0]:
                s, a = state.east()
            elif dest[1] < state.pos[1]:
                s, a = state.down()
            elif dest[1] > state.pos[1]:
                s, a = state.up()
            elif dest[2] < state.pos[2]:
                s, a = state.south()
            elif dest[2] > state.pos[2]:
                s, a = state.north()
            else:
                assert False
            demo.append((state, a, s))
            state = s
        return demo
