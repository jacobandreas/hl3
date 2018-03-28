#!/usr/bin/env python3

from craft.builder import Block, BlockType, Scene, House, Window, Door, Wall, Course, Tree
from craft.builder import BuilderException

from collections import Counter, namedtuple
import gflags
import numpy as np
from skimage.measure import block_reduce
from skimage.transform import downscale_local_mean

FLAGS = gflags.FLAGS

def dist(pos1, pos2):
    assert len(pos1) == len(pos2)
    return sum(abs(q1 - q2) for q1, q2 in zip(pos1, pos2))

class CraftEnv(object):
    GO = 0
    ADD = 1
    REMOVE = 2
    CLONE = 3
    STOP = 4
    SAY = 5
    n_actions = 6
    
    world_shape = Scene._size
    n_block_types = 1 + len(BlockType.enumerate())
    n_world_obs = n_block_types + 1
    n_state_obs = n_block_types + (5 * 5 * 5 * n_block_types)

    _action_names = {
            GO: 'g',
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
                if FLAGS.debug and task.desc not in self.ALLOWED:
                    continue
                break
            except BuilderException as e:
                print(e)
                pass
        return task

    @classmethod
    def action_name(cls, action):
        return cls._action_names[action]

class CraftState(namedtuple('CraftState', ['blocks', 'pos', 'mat'])):
    @classmethod
    def from_scene(cls, scene, pos, mat):
        blocks = np.zeros((CraftEnv.n_block_types,) + CraftEnv.world_shape)
        for block in scene.blocks():
            blocks[(block.block_type.mat_id(),) + block.pos] = 1
        return CraftState(blocks, pos, mat)

    def to_scene(self):
        size = self.blocks.shape[1:]
        occupied = (self.blocks.argmax(axis=0) > 0).astype(int)

        parts = []
        for x in range(size[0]):
            for y in range(size[1]):
                for z in range(size[2]):
                    if not occupied[x, y, z]:
                        continue
                    parts.append(Block(
                        (x, y, z),
                        BlockType.with_id(self.blocks[:, x, y, z].argmax())))

        return Scene(size, parts, occupied)

    def go(self, pos):
        assert (len(pos) == 3 
            and all(0 <= p < l for p, l in zip(pos, self.blocks.shape[1:])))
        return self._replace(pos=pos), (CraftEnv.GO, pos)

    def clone(self):
        x, y, z = self.pos
        if not self.blocks[:, x, y, z].any():
            return self, CraftEnv.CLONE
        return self._replace(mat=self.blocks[:, x, y, z].argmax()), (CraftEnv.CLONE, None)

    def add(self):
        x, y, z = self.pos
        if self.blocks[:, x, y, z].any():
            return self, CraftEnv.ADD
        blocks = self.blocks.copy()
        blocks[self.mat, x, y, z] = 1
        return self._replace(blocks=blocks), (CraftEnv.ADD, None)

    def remove(self):
        x, y, z = self.pos
        if not self.blocks[:, x, y, z].any():
            return self, CraftEnv.REMOVE
        blocks = self.blocks.copy()
        blocks[:, x, y, z] = 0
        return self._replace(blocks=blocks), (CraftEnv.REMOVE, None)

    # TODO REFACTOR!!!
    def step(self, full_action):
        action, pos = full_action
        if action == CraftEnv.GO:
            return self.go(pos)[0]
        elif action == CraftEnv.CLONE:
            return self.clone()[0]
        elif action == CraftEnv.ADD:
            return self.add()[0]
        elif action == CraftEnv.REMOVE:
            return self.remove()[0]
        elif action == CraftEnv.STOP:
            return None
        elif action == CraftEnv.SAY:
            return self
        else:
            assert False, "unknown action %d" % action

    def obs(self):
        pos_features = np.zeros((1,) + CraftEnv.world_shape)
        x, y, z = self.pos
        pos_features[0, x, y, z] = 1
        world_features = np.concatenate((self.blocks, pos_features), axis=0)

        def pad_slice(f, w, h, d, x, y, z, data):
            e = np.zeros((f, w+4, h+4, d+4))
            e[:, 2:-2, 2:-2, 2:-2] = data
            ft = e[:, x-2+2:x+3+2, y-2+2:y+3+2, z-2+2:z+3+2]
            assert ft.shape[1:] == (5, 5, 5), \
                "bad slice with %d %d %d / %d %d %d" % (w, h, d, x, y, z)
            return ft

        f = CraftEnv.n_block_types
        w, h, d = CraftEnv.world_shape
        local_features = pad_slice(f, w, h, d, x, y, z, self.blocks)
        mat_features = np.zeros((CraftEnv.n_block_types,))
        mat_features[self.mat] = 1
        state_features = np.concatenate((
            local_features.ravel(), mat_features))

        return state_features, world_features

class Task(object):
    FIND = 0
    ADD = 1
    REMOVE = 2
    CLONE = 3

    _actions = [FIND, ADD, REMOVE, CLONE]
    _action_probs = [0.3, 0.3, 0.3, 0.1]
    #_action_probs = [0.8, 0., 0., 0.2]
    _action_names = {
        FIND: 'find',
        ADD: 'add',
        REMOVE: 'remove',
        CLONE: 'clone'
    }

    @classmethod
    def sample(cls):
        scene1 = Scene.sample()
        action = np.random.choice(cls._actions, p=cls._action_probs)

        parts = list(scene1.parts())
        def part_filter(part):
            if action == cls.ADD and isinstance(part, Block):
                return False
            if action == cls.ADD and isinstance(part, Wall) and part.incomplete:
                return False
            if action == cls.ADD or action == cls.REMOVE:
                return (
                    not isinstance(part, House) 
                    and not isinstance(part, Tree))
            if action == cls.CLONE:
                return (
                    not isinstance(part, Window) 
                    and not isinstance(part, Door))
            return True
        parts = [p for p in parts if part_filter(p)]
        if len(parts) == 0:
            assert False
        part = parts[np.random.randint(len(parts))]
        scene2 = scene1.remove(part)

        descs = list(part.descriptions(top=True))
        here = action in (cls.ADD, cls.REMOVE) and np.random.random() < 0.25
        if here:
            desc = descs[0]
        else:
            descs = list(set(descs))
            desc = descs[np.random.randint(len(descs))]

        if action == cls.FIND:
            scene_before = scene_after = scene1
        elif action == cls.REMOVE:
            scene_before = scene1
            scene_after = scene2
        elif action == cls.ADD:
            scene_before = scene2
            scene_after = scene1
        elif action == cls.CLONE:
            scene_before = scene_after = scene1
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

    def demonstration(self):
        if self.action == self.FIND:
            demo, state = self._demonstrate_find(self.init_state)
        elif self.action == self.CLONE:
            demo, state = self._get_mat(self.init_state, self.part.block_type)
        else:
            demo, state = self._demonstrate_change(self.init_state)
        demo.append((state, (CraftEnv.STOP, None), None))
        assert self.validate(demo[-1][0], debug=False) > 0, self.desc
        return demo

    def _demonstrate_find(self, state):
        goal_opts = list(self.part.positions())
        goal = goal_opts[np.random.randint(len(goal_opts))]
        return self._go_to(state, goal)

    def _demonstrate_change(self, state):
        blocks_before = set(self.scene_before.blocks())
        blocks_after = set(self.scene_after.blocks())
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
            assert state_with.blocks[:, nx, ny, nz].sum() > 0
            assert state_without.blocks[:, nx, ny, nz].sum() == 0

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
        if state.pos == dest:
            demo = []
        else:
            s_, a = state.go(dest)
            demo = [(state, a, s_)]
            state = s_
        return demo, state

    def validate(self, state, debug=False):
        init_pos = self.init_state.pos
        final_pos = state.pos

        if debug:
            px, _, pz = final_pos
            px1, pz1 = max(px-2, 0), max(pz-2, 0)
            px2, pz2 = px+2, pz+2
            sl = state.blocks[:, px1:px2+1, :, pz1:pz2+1].sum(axis=(0, 2))

            goal_pos = self.demonstration()[-1][0].pos
            gx, _, gz = goal_pos
            gx1, gz1 = max(gx-2, 0), max(gz-2, 0)
            gx2, gz2 = gx+2, gz+2
            gl = state.blocks[:, gx1:gx2+1, :, gz1:gz2+1].sum(axis=(0, 2))
            #print("sl")
            #print(sl)
            #print("gl")
            #print(gl)
            #print("everything")
            #print(state.blocks.sum(axis=(0, 2)))

        if self.action == self.FIND:
            return self._validate_find(final_pos, debug=debug)
        elif self.action == self.CLONE:
            return self._validate_clone(state.mat, debug=debug)
        elif self.action == self.ADD:
            return self._validate_add(state)
        elif self.action == self.REMOVE:
            return self._validate_remove(state)
        else:
            return 1

    def _validate_find(self, final_pos, debug=True):
        parts = [part for part in self.scene_before.parts() if final_pos in part.positions()]
        if len(parts) == 0:
            return 0.
        descs = [desc for part in parts for desc in part.descriptions(top=True)]
        if(debug):
            print(descs)
        if self._part_desc in descs:
            return 1.
        return 0.

    def _validate_clone(self, final_mat, debug=True):
        if debug:
            print(BlockType.with_id(final_mat).material)
        if final_mat == self.part.block_type.mat_id():
            return 1.
        return 0.

    def _get_delta(self, state):
        parts = [
            p for p in self.scene_before.parts()
            if isinstance(p, type(self.part))]
        parts = [
            p for p in parts
            if self._part_desc in p.descriptions(top=True)]

        init_blocks = self.init_state.blocks.sum(axis=0)
        final_blocks = state.blocks.sum(axis=0)

        added = np.where(final_blocks > init_blocks)
        removed = np.where(init_blocks > final_blocks)

        if len(added[0]) == 0:
            added_shape = added_missing = None
        else:
            added_bbox = ([min(a) for a in added], [max(a)+1 for a in added])
            ((x1, y1, z1), (x2, y2, z2)) = added_bbox
            box_blocks = final_blocks[x1:x2, y1:y2, z1:z2]
            added_missing = box_blocks.size - box_blocks.sum()
            added_shape = (x2-x1, y2-y1, z2-z1)
        if len(removed[0]) == 0:
            removed_shape = removed_missing = None
        else:
            removed_bbox = ([min(r) for r in removed], [max(r)+1 for r in removed])
            ((x1, y1, z1), (x2, y2, z2)) = removed_bbox
            box_blocks = final_blocks[x1:x2, y1:y2, z1:z2]
            removed_missing = box_blocks.sum()
            removed_shape = (x2-x1, y2-y1, z2-z1)

        return added_shape, added_missing, removed_shape, removed_missing, parts

        #a_x, a_y, a_z = added
        #a_x, a_y, a_z = set(a_x), set(a_y), set(a_z)
        #added_distinct = sorted([len(a_x), len(a_y), len(a_z)])
        #r_x, r_y, r_z = removed
        #r_x, r_y, r_z = set(r_x), set(r_y), set(r_z)
        #removed_distinct = sorted([len(r_x), len(r_y), len(r_z)])

        #return added_distinct, removed_distinct, parts

    def _validate_add(self, state):
        added_shape, added_missing, removed_shape, removed_missing, _ = self._get_delta(state)

        # TODO actually located in wall
        if isinstance(self.part, Window):
            return float(
                added_shape is None
                and removed_shape is not None
                and removed_missing == 0
                and removed_shape == (1, 1, 1))

        # TODO actually located in wall
        elif isinstance(self.part, Door):
            return float(
                added_shape is None
                and removed_shape is not None
                and removed_missing == 0
                and removed_shape == (1, 2, 1))

        # TODO validate block type
        elif isinstance(self.part, Course):
            return float(
                removed_shape is None
                and added_shape is not None
                and added_missing <= 1
                and added_shape[1] == 1
                and ((added_shape[0] == 1 and added_shape[2] > 1)
                     or (added_shape[2] == 1 and added_shape[0] > 1)))

        elif isinstance(self.part, Wall):
            return float(
                removed_shape is None
                and added_shape is not None
                and added_missing <= 3
                and added_shape[1] >= 1
                and ((added_shape[0] == 1 and added_shape[2] > 1)
                     or (added_shape[2] == 1 and added_shape[0] > 1)))

        print(self.desc)
        assert False

    def _validate_remove(self, state):
        added_shape, added_missing, removed_shape, removed_missing, candidates = self._get_delta(state)

        if isinstance(self.part, Block):
            removed = [
                not state.blocks[:, p.pos[0], p.pos[1], p.pos[2]].any()
                for p in candidates]
            return float(
                added_shape is None
                and removed_shape is not None
                and removed_shape == (1, 1, 1)
                and any(removed))

        if isinstance(self.part, Window) or isinstance(self.part, Door):
            filled = []
            for p in candidates:
                # TODO global property of wall
                mat = next(p.parent[0].blocks()).block_type.mat_id()
                filled.append(all(
                    state.blocks[mat, x, y, z] == 1
                    for x, y, z in p.positions()))
            return float(
                removed_shape is None
                and any(filled))

        if isinstance(self.part, Course) or isinstance(self.part, Wall):
            removed = []
            for p in candidates:
                removed.append(all(
                    not state.blocks[:, x, y, z].any()
                    for x, y, z in p.positions()))
            return float(
                added_shape is None
                and any(removed))

        print(self.desc)
        assert False
