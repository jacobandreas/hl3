#!/usr/bin/env python3

from craft.builder import Block, BlockType, Scene, House, Window, Door, Wall, Course

from collections import Counter, namedtuple
import numpy as np
from skimage.measure import block_reduce
from skimage.transform import downscale_local_mean

def dist(pos1, pos2):
    assert len(pos1) == len(pos2)
    return sum(abs(q1 - q2) for q1, q2 in zip(pos1, pos2))

class CraftEnv(object):
    WEST = 0
    EAST = 1
    DOWN = 2
    UP = 3
    SOUTH = 4
    NORTH = 5
    ADD = 6
    REMOVE = 7
    CLONE = 8
    STOP = 9
    n_actions = 10
    n_features = 5 * 5 * 5 * 7 * 3 + 3 + (1 + len(BlockType.enumerate()))

    _action_names = {
        WEST: 'w',
        EAST: 'e',
        DOWN: 'd',
        UP: 'u',
        SOUTH: 's',
        NORTH: 'n',
        ADD: 'A',
        REMOVE: 'R',
        CLONE: 'C',
        STOP: '.',
    }

    ALLOWED = set([
        'add a wood course',
        'clone a wood block',
        'add a course'])

    @classmethod
    def sample_task(self):
        while True:
            try:
                task = Task.sample()
                #if task.desc not in self.ALLOWED:
                #    continue
                break
            except AssertionError as e:
                pass
        return task

    @classmethod
    def action_name(cls, action):
        return cls._action_names[action]

class CraftState(namedtuple('CraftState', ['blocks', 'pos', 'mat'])):
    @classmethod
    def from_scene(cls, scene, pos, mat):
        blocks = np.zeros(scene.size + (len(BlockType.enumerate()) + 1,))
        for block in scene.blocks():
            blocks[block.pos + (block.block_type.mat_id(),)] = 1
        #if mat is None:
        #    mat = len(BlockType.enumerate()) + 1
        return CraftState(blocks, pos, mat)

    def to_scene(self):
        size = self.blocks.shape[:3]
        occupied = (self.blocks.argmax(axis=3) > 0).astype(int)

        parts = []
        for x in range(size[0]):
            for y in range(size[1]):
                for z in range(size[2]):
                    if not occupied[x, y, z]:
                        continue
                    parts.append(Block(
                        (x, y, z),
                        BlockType.with_id(self.blocks[x, y, z, :].argmax())))

        return Scene(size, parts, occupied)


    def move(self, delta, act):
        npos = tuple(p + d for p, d in zip(self.pos, delta))
        ok = all(n >= 0 for n in npos) and all(n < s for n, s in zip(npos, self.blocks.shape))
        if not ok:
            return self, act
        return self._replace(pos=npos), act

    def west(self):
        return self.move((-1, 0, 0), CraftEnv.WEST)

    def east(self):
        return self.move((1, 0, 0), CraftEnv.EAST)

    def down(self):
        return self.move((0, -1, 0), CraftEnv.DOWN)

    def up(self):
        return self.move((0, 1, 0), CraftEnv.UP)

    def south(self):
        return self.move((0, 0, -1), CraftEnv.SOUTH)

    def north(self):
        return self.move((0, 0, 1), CraftEnv.NORTH)

    def clone(self):
        if not self.blocks[self.pos].any():
            return self, CraftEnv.CLONE
        return self._replace(mat=self.blocks[self.pos].argmax()), CraftEnv.CLONE

    def add(self):
        if self.blocks[self.pos].any():
            return self, CraftEnv.ADD
        blocks = self.blocks.copy()
        blocks[self.pos + (self.mat,)] = 1
        return self._replace(blocks=blocks), CraftEnv.ADD

    def remove(self):
        if not self.blocks[self.pos].any():
            return self, CraftEnv.REMOVE
        blocks = self.blocks.copy()
        blocks[self.pos] = 0
        return self._replace(blocks=blocks), CraftEnv.REMOVE

    # TODO REFACTOR!!!
    def step(self, action):
        if action == CraftEnv.WEST:
            return self.west()[0]
        elif action == CraftEnv.EAST:
            return self.east()[0]
        elif action == CraftEnv.DOWN:
            return self.down()[0]
        elif action == CraftEnv.UP:
            return self.up()[0]
        elif action == CraftEnv.SOUTH:
            return self.south()[0]
        elif action == CraftEnv.NORTH:
            return self.north()[0]
        elif action == CraftEnv.CLONE:
            return self.clone()[0]
        elif action == CraftEnv.ADD:
            return self.add()[0]
        elif action == CraftEnv.REMOVE:
            return self.remove()[0]
        elif action == CraftEnv.STOP:
            return None
        else:
            assert False, "unknown action %d" % action

    def _block_features(self, blocks):
        from math import ceil, floor

        def pad_slice(w, h, d, f, x, y, z, data):
            assert data.shape == (w, h, d, f)
            e = np.zeros((w+4, h+4, d+4, f))
            e[2:-2, 2:-2, 2:-2, :] = data
            ft = e[x-2+2:x+3+2, y-2+2:y+3+2, z-2+2:z+3+2, :]
            assert ft.shape[:3] == (5, 5, 5), \
                "bad slice with %d %d %d / %d %d %d" % (w, h, d, x, y, z)
            return ft

        x, y, z = self.pos

        p0 = blocks
        w, h, d, f = p0.shape
        f0 = pad_slice(w, h, d, f, x, y, z, p0)

        w1, h1, d1 = ceil(w/3), ceil(h/3), ceil(d/3)
        x1, y1, z1 = floor(x/3), floor(y/3), floor(z/3)
        #p1 = block_reduce(p0, (3, 3, 3, 1), func=np.mean)
        p1 = downscale_local_mean(p0, (3, 3, 3, 1))
        f1 = pad_slice(w1, h1, d1, f, x1, y1, z1, p1)

        w2, h2, d2 = ceil(w1/3), ceil(h1/3), ceil(d1/3)
        x2, y2, z2 = floor(x1/3), floor(y1/3), floor(z1/3)
        #p2 = block_reduce(p1, (3, 3, 3, 1), func=np.mean)
        p2 = downscale_local_mean(p1, (3, 3, 3, 1))
        f2 = pad_slice(w2, h2, d2, f, x2, y2, z2, p2)

        return np.concatenate((f0.ravel(), f1.ravel(), f2.ravel()))

    def features(self):
        mat_features = np.zeros((1 + len(BlockType.enumerate()),))
        mat_features[self.mat] = 1
        return np.concatenate((
            self._block_features(self.blocks),
            mat_features,
            np.asarray(self.pos) / np.asarray(self.blocks.shape[:3])))

class Task(object):
    FIND = 0
    ADD = 1
    REMOVE = 2
    CLONE = 3

    #_actions = [FIND, ADD, REMOVE]
    _actions = [FIND, ADD, REMOVE, CLONE]
    _action_probs = [0.3, 0.3, 0.3, 0.1]
    _action_names = {
        FIND: 'find',
        ADD: 'add',
        REMOVE: 'remove',
        CLONE: 'clone'
    }

    @classmethod
    def sample(cls):
        scene1 = Scene.sample()
        parts = list(scene1.parts())
        #part_filter = lambda x: not (isinstance(x, House) or isinstance(x,
        #    Window) or isinstance(x, Door))
        part_filter = lambda x: not isinstance(x, House)
        parts = [p for p in parts if part_filter(p)]
        if len(parts) == 0:
            assert False
        part = parts[np.random.randint(len(parts))]
        scene2 = scene1.remove(part)
        action = np.random.choice(cls._actions, p=cls._action_probs)

        descs = list(part.descriptions(top=True))
        here = action in (cls.ADD, cls.REMOVE) and np.random.random() < 0.25
        if here:
            desc = descs[0]
        else:
            descs = list(set(descs))
            desc = descs[np.random.randint(len(descs))]

        if action == cls.FIND:
            scene_before = scene_after = scene1
            assert not here
        elif action == cls.REMOVE:
            scene_before = scene1
            scene_after = scene2
        elif action == cls.ADD:
            scene_before = scene2
            scene_after = scene1
        elif action == cls.CLONE:
            scene_before = scene_after = scene1
            assert hasattr(part, 'blocks')
            assert not here
            blocks = list(part.blocks())
            part = blocks[np.random.randint(len(blocks))]
            desc = 'a ' + next(part.block_type.descriptions()) + ' block'

        return Task(action, part, desc, scene_before, scene_after, here)

    def __init__(self, action, part, part_desc, scene_before, scene_after, here):
        self.action = action
        self.part = part
        self._part_desc = part_desc
        self.here = here
        self.desc = next(self._descriptions())
        self.scene_before = scene_before
        self.scene_after = scene_after
        self.here = here

        init_pos = [np.random.randint(dim) for dim in self.scene_before.size]
        mat_ids = [b.mat_id() for b in BlockType.enumerate()]
        init_mat = np.random.choice(mat_ids)
        self.init_state = CraftState.from_scene(self.scene_before,
                tuple(init_pos), init_mat)
        if here:
            demo = self.demonstration()
            after_clone = [s_ for s, a, s_ in demo if a == CraftEnv.CLONE]
            init_pos = part.pos
            if len(after_clone) > 0:
                init_mat = after_clone[0].mat
        self.init_state = CraftState.from_scene(self.scene_before,
                tuple(init_pos), init_mat)

    def _descriptions(self):
        here = ' here' if self.here else ''
        yield '%s %s%s' % (self._action_names[self.action], self._part_desc, here)

    def dump(self):
        with open('../vis/scene.json', 'w') as scene_f:
            self.scene_before.dump(scene_f)

    def demonstration(self):
        if self.action == self.FIND:
            demo, state = self._demonstrate_find(self.init_state)
        elif self.action == self.CLONE:
            demo, state = self._get_mat(self.init_state, self.part.block_type)
        else:
            demo, state = self._demonstrate_change(self.init_state)
        demo.append((state, CraftEnv.STOP, None))
        return demo

    def _demonstrate_find(self, state):
        return self._go_to(state, self.part.pos)

    def _demonstrate_change(self, state):
        blocks_before = set(self.scene_before.blocks())
        blocks_after = set(self.scene_after.blocks())
        #to_add = blocks_after - blocks_before
        #to_remove = blocks_before - blocks_after
        to_add = [b for b in self.scene_after.blocks() if b not in blocks_before]
        to_remove = [b for b in self.scene_before.blocks() if b not in blocks_after]

        if len(to_remove) == 0:
            remaining = list(reversed(to_add))
            add = True
        elif len(to_add) == 0:
            remaining = to_remove
            add = False
        else:
            assert False, ("to add", to_add, "to remove", to_remove)

        demo = []

        build_order = remaining
        #build_order = list(reversed(sorted(remaining, 
        #        key=lambda x: (x.block_type.mat_id(), dist(x.pos, state.pos)))))

        while len(build_order) > 0:
            nearest = build_order.pop()
            if add and state.mat != nearest.block_type.mat_id():
                ndemo, state = self._get_mat(state, nearest.block_type)
                demo += ndemo
            ndemo, state = self._go_to(state, nearest.pos)
            demo += ndemo
            if add:
                s_, a = state.add()
                state_without = state
                state_with = s_
            else:
                s_, a = state.remove()
                state_with = state
                state_without = s_

            nx, ny, nz = nearest.pos
            assert state_with.blocks[nx, ny, nz, :].sum() > 0
            assert state_without.blocks[nx, ny, nz, :].sum() == 0

            demo.append((state, a, s_))
            state = s_
        return demo, state

    def _get_mat(self, state, block_type):
        blocks_before = self.scene_before.blocks()
        exemplars = [b for b in blocks_before if b.block_type == block_type]
        nearest = min(exemplars, key=lambda x: dist(x.pos, state.pos))
        demo, state = self._go_to(state, nearest.pos)
        s_, a = state.clone()
        demo.append((state, a, s_))
        state = s_
        return demo, state

    def _go_to(self, state, dest):
        demo = []
        while state.pos != dest:
            if dest[0] < state.pos[0]:
                s_, a = state.west()
            elif dest[0] > state.pos[0]:
                s_, a = state.east()
            elif dest[1] < state.pos[1]:
                s_, a = state.down()
            elif dest[1] > state.pos[1]:
                s_, a = state.up()
            elif dest[2] < state.pos[2]:
                s_, a = state.south()
            elif dest[2] > state.pos[2]:
                s_, a = state.north()
            else:
                assert False
            demo.append((state, a, s_))
            state = s_
        return demo, state
