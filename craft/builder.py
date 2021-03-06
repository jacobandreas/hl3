from collections import namedtuple
import json
import numpy as np

MAX_TRIES = 20

class BuilderException(Exception):
    def __init__(self, msg):
        super().__init__(msg)

class BlockType(namedtuple('BlockType', ['material'])):
    _materials = [
        'grass',
        'red_stone',
        'wood',
        'blue_stone',
        'clear_glass',
        'blue_glass'
    ]

    _descriptions = {
        'grass': 'grass',
        'red_stone': 'red',
        'wood': 'wood',
        'blue_stone': 'blue',
        'clear_glass': 'clear',
        'blue_glass': 'blue'
    }
    
    def __new__(cls, material):
        assert material in cls._materials
        return super().__new__(cls, material)

    @classmethod
    def enumerate(cls):
        return [BlockType(m) for m in cls._materials]

    def mat_id(self):
        return self._materials.index(self.material) + 1

    @classmethod
    def with_id(cls, mat_id):
        return BlockType(cls._materials[mat_id-1])

    def descriptions(self, mentioned=[]):
        yield self._descriptions[self.material]

class Tree(namedtuple('Tree', ['pos', 'leaf_type', 'kind', 'tree_blocks'])):
    SHORT = 0
    TALL = 1

    _heights = {
        SHORT: 5,
        TALL: 8
    }
    _kind_names = {
        SHORT: 'short',
        TALL: 'tall'
    }

    _branch_height = 3

    def __repr__(self):
        return 'Tree(%r, %r, %r)' % (self.pos, self.leaf_type, self.kind)

    @classmethod
    def sample(cls, pos):
        x, y, z = pos
        trunk_type = BlockType('wood')
        leaf_type = BlockType('grass')
        kind = np.random.randint(2)
        blocks = []
        for h in range(cls._heights[kind]):
            by = y + h
            block_type = trunk_type if (h < cls._heights[kind] - 1) else leaf_type
            block = Block((x, by, z), block_type)
            blocks.append(block)

            if h >= cls._branch_height:
                for ox, oz in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if np.random.random() < 0.4:
                        continue
                    lx = x + ox
                    lz = z + oz
                    block = Block((lx, by, lz), leaf_type)
                    blocks.append(block)

        return Tree(pos, leaf_type, kind, blocks)

    def blocks(self):
        for block in self.tree_blocks:
            yield block

    def positions(self):
        for block in self.blocks():
            yield block.pos

    def parts(self):
        yield self

    def remove(self, rem):
        return self, 0

    def descriptions(self, top=False, mentioned=[]):
        yield 'a tree'
        yield 'a %s tree' % self._kind_names[self.kind]

class Block(namedtuple('Block', ['pos', 'block_type'])):
    def descriptions(self, top=False, mentioned=[]):
        yield 'a block'
        for desc in self.block_type.descriptions():
            yield 'a %s block' % desc

    def positions(self):
        return {self.pos}

    def blocks(self):
        yield self

class Window(namedtuple('Window', ['pos', 'parent'])):
    def descriptions(self, top=False, mentioned=[]):
        yield 'a window'
        if top:
            for desc in self.parent[0].descriptions(mentioned=mentioned+[self]):
                yield 'a window in %s' % desc

    def positions(self):
        return {self.pos}

class Door(namedtuple('Door', ['pos', 'parent'])):
    def descriptions(self, top=False, mentioned=[]):
        yield 'a door'
        if top:
            for desc in self.parent[0].descriptions(mentioned=mentioned+[self]):
                yield 'a door in %s' % desc

    def positions(self):
        dx, dy, dz = self.pos
        return {self.pos, (dx, dy+1, dz)}
        
class Roof(namedtuple('Roof', ['pos', 'roof_blocks'])):
    _block_types = [
        BlockType('wood'),
    ]

    def __repr__(self):
        return 'Roof(pos=%r)' % (self.pos,)

    @classmethod
    def sample(cls, pos, width, height, depth):
        x, y, z = pos
        block_type = cls._block_types[np.random.choice(len(cls._block_types))]
        blocks = []
        for w in range(1, width - 1):
            for d in range(1, depth - 1):
                bx = x + w
                by = y + height - 1
                bz = z + d
                block = Block((bx, by, bz), block_type)
                blocks.append(block)
        return Roof(pos, blocks)

    def blocks(self):
        for block in self.roof_blocks:
            yield block

    def positions(self):
        for block in self.blocks():
            yield block.pos

class Course(namedtuple('Course', ['pos', 'block_type', 'course_blocks'])):
    def __repr__(self):
        return 'Course(pos=%r, block_type=%r)'

    @classmethod
    def sample(cls, pos, angle, length, block_type):
        x, y, z = pos
        blocks = []
        for l in range(length):
            if angle == Wall.NS:
                bx = x
                bz = z + l
            elif angle == Wall.EW:
                bx = x + l
                bz = z
            block = Block((bx, y, bz), block_type)
            blocks.append(block)
        return Course(pos, block_type, blocks)

    def descriptions(self, top=False, mentioned=[]):
        yield "a course"
        for desc in self.block_type.descriptions():
            yield "a %s course" % desc

    def blocks(self):
        for block in self.course_blocks:
            yield block

    def positions(self):
        for block in self.blocks():
            yield block.pos

class Wall(namedtuple('Wall', ['pos', 'block_type', 'courses', 'window', 'door', 'height', 'incomplete'])):
    NS = 0
    EW = 1

    _max_windows = 1
    _max_doors = 1
    _block_types = [
        BlockType('red_stone'),
        BlockType('wood'),
        BlockType('blue_stone')
    ]

    def __repr__(self):
        return 'Wall(pos=%r, block_type=%r, window=%r, door=%r, height=%r, inc=%r)' % (
                self.pos, self.block_type, self.window, self.door, self.height,
                self.incomplete)

    @classmethod
    def sample(cls, pos, angle, length, height, block_type=None, incomplete=False):
        x, y, z = pos
        if block_type is None:
            block_type = cls._block_types[np.random.choice(len(cls._block_types))]
        n_windows = np.random.randint(cls._max_windows + 1)
        n_doors = np.random.randint(cls._max_doors + 1)

        course_height = height
        if incomplete:
            course_height = np.random.randint(1, height)

        if angle == Wall.NS:
            window_x = x
            window_z = z + length - 2
            door_x = x
            door_z = z + length // 2
        elif angle == Wall.EW:
            window_x = x + length - 2
            window_z = z
            door_x = x + length // 2
            door_z = z

        window = None
        door = None
        if not incomplete:
            if n_windows == 1:
                window = Window((window_x, y+height-2, window_z), [])
            if n_doors == 1:
                door = Door((door_x, y, door_z), [])

        courses = []
        for h in range(course_height):
            course = Course.sample((x, y+h, z), angle, length, block_type)
            courses.append(course)
        #blocks = []
        #for l in range(length):
        #    for h in range(height):
        #        if window is not None and l == length - 2 and h == height - 2:
        #            continue
        #        if door is not None and l == length // 2 and h < 2:
        #            continue
        #        by = y + h
        #        if angle == Wall.NS:
        #            bx = x
        #            bz = z + l
        #        elif angle == Wall.EW:
        #            bx = x + l
        #            bz = z
        #        block = Block((bx, by, bz), block_type)
        #        blocks.append(block)
        #wall = Wall(pos, block_type, tuple(blocks), window, door, height)
        wall = Wall(pos, block_type, tuple(courses), window, door, course_height, incomplete)

        # TODO YIKES
        if window is not None:
            window.parent.append(wall)
        if door is not None:
            door.parent.append(wall)

        return wall

    def blocks(self):
        for course in self.courses:
            for block in course.course_blocks:
                if self.window is not None and block.pos in self.window.positions():
                    continue
                if self.door is not None and block.pos in self.door.positions():
                    continue
                yield block

    def positions(self):
        for block in self.blocks():
            yield block.pos

    def parts(self):
        yield self
        if len(self.courses) > 0:
            yield self.courses[-1]
        if self.door is not None:
            yield self.door
        if self.window is not None:
            yield self.window

    def remove(self, rem):
        count = 0
        window = self.window
        door = self.door
        ncourses = self.courses
        if self.window is not None and isinstance(rem, Window) and (rem == self.window):
            count += 1
            wx, wy, wz = self.window.pos
            window = None
        if self.door is not None and isinstance(rem, Door) and (rem == self.door):
            count += 1
            dx, dy, dz = self.door.pos
            door = None
        if isinstance(rem, Course) and rem == self.courses[-1]:
            ncourses = self.courses[:-1]
            count += 1
        return Wall(self.pos, self.block_type, ncourses, window, door,
                self.height, self.incomplete), count

    def descriptions(self, top=False, mentioned=[]):
        def self_descs():
            yield 'a wall'
            for desc in self.block_type.descriptions():
                yield 'a %s wall' % desc
        for desc in self_descs():
            yield desc
            if self.incomplete:
                continue
            if self.window is not None and self.window not in mentioned:
                for w_desc in self.window.descriptions(mentioned=mentioned+[self]):
                    yield '%s with %s' % (desc, w_desc)
            if self.door is not None and self.door not in mentioned:
                for d_desc in self.door.descriptions(mentioned=mentioned+[self]):
                    yield '%s with %s' % (desc, d_desc)

class House(namedtuple('House', ['pos', 'walls', 'roof', 'block_type'])):
    _height = 5

    _block_types = [
        BlockType('red_stone'),
        BlockType('wood'),
        BlockType('blue_stone'),
        None,
        None
    ]

    @classmethod
    def sample(cls, scene):
        s_width, s_height, s_depth = scene.size
        width = np.random.randint(5, 7)
        depth = np.random.randint(5, 7)
        block_type = cls._block_types[np.random.randint(len(cls._block_types))]
        x = np.random.randint(s_width - width)
        y = 0
        z = np.random.randint(s_depth - depth)
        walls = [
            Wall.sample((x, y, z), Wall.EW, width, cls._height,
                block_type=block_type),
            Wall.sample((x, y, z + depth - 1), Wall.EW, width, cls._height,
                block_type=block_type),

            Wall.sample((x, y, z + 1), Wall.NS, depth - 2, cls._height,
                block_type=block_type),
            Wall.sample((x + width - 1, y, z + 1), Wall.NS, depth - 2, cls._height,
                block_type=block_type)
        ]
        roof = Roof.sample((x, y, z), width, cls._height, depth)
        return House((x, y, z), walls, roof, block_type)

    def blocks(self):
        for wall in self.walls:
            for block in wall.blocks():
                yield block
        for block in self.roof.blocks():
            yield block

    def parts(self):
        yield self
        for wall in self.walls:
            for part in wall.parts():
                yield part

    def remove(self, rem):
        assert rem != self.roof
        walls = []
        count = 0
        for wall in self.walls:
            if wall == rem:
                count += 1
            else:
                nwall, ncount = wall.remove(rem)
                count += ncount
                walls.append(nwall)
        return House(self.pos, walls, self.roof, self.block_type), count

    def descriptions(self, top=False, mentioned=[]):
        yield 'a house'
        if self.block_type is not None:
            for desc in self.block_type.descriptions(mentioned=mentioned+[self]):
                yield 'a %s house' % desc
        for part in self.parts():
            if part is self or part in mentioned:
                continue
            for desc in part.descriptions(mentioned=mentioned+[self]):
                yield 'a house with %s' % desc

class Scene(object):
    #_max_houses = 2
    #_max_walls = 3
    #_max_trees = 4

    _max_houses = 0
    _max_walls = 3
    _max_trees = 4

    _size = (25, 10, 25)

    @classmethod
    def sample(cls):
        n_houses = 0 if cls._max_houses == 0 else np.random.randint(cls._max_houses + 1)
        n_walls = 0 if cls._max_walls == 0 else (
            (0 if n_houses >= 1 else 1) + np.random.randint(cls._max_walls))
        n_trees = 0 if cls._max_trees == 0 else np.random.randint(cls._max_trees)

        scene = cls.empty()
        for _ in range(n_houses):
            scene = scene.add_house()
        scene = scene.add_walls(n_walls)
        for _ in range(n_trees):
            scene = scene.add_tree()
        for block_type in BlockType.enumerate():
            scene = scene.add_block(block_type)
        return scene

    @classmethod
    def empty(cls):
        return Scene(cls._size, (), np.zeros(cls._size, dtype=np.bool))

    def __init__(self, size, parts, occupied):
        self.size = size
        self._parts = parts
        self.occupied = occupied

    def add_house(self):
        for _ in range(MAX_TRIES):
            house = House.sample(self)
            positions = list(block.pos for block in house.blocks())
            if any(self.occupied[pos] for pos in positions):
                continue
            parts = self._parts + (house,)
            occupied = self.occupied.copy()
            for pos in positions:
                occupied[pos] = 1
            return Scene(self.size, parts, occupied)

    def add_walls(self, n_walls):
        if n_walls == 0:
            return self
        for _ in range(MAX_TRIES):
            s_width, s_height, s_depth = self.size
            width = 3 + np.random.randint(5)
            depth = 5 + np.random.randint(5)
            height = House._height
            assert width > 0 and depth > 0
            x = np.random.randint(s_width - width)
            y = 0
            z = np.random.randint(s_depth - depth)
            incomplete = np.random.randint(2, size=4)
            walls = [
                Wall.sample((x, y, z), Wall.EW, width, height,
                    incomplete=incomplete[0]),
                Wall.sample((x, y, z + depth - 1), Wall.EW, width, height,
                    incomplete=incomplete[1]),
                Wall.sample((x, y, z + 1), Wall.NS, depth - 2, height,
                    incomplete=incomplete[2]),
                Wall.sample((x + width - 1, y, z + 1), Wall.NS, depth - 2, height,
                    incomplete=incomplete[3])
            ]
            wall_order = np.random.shuffle(walls)
            walls = walls[:n_walls]
            positions = [block.pos for wall in walls for block in wall.blocks()]
            if any(self.occupied[pos] for pos in positions):
                continue
            parts = self._parts + tuple(walls)
            occupied = self.occupied.copy()
            for pos in positions:
                occupied[pos] = 1
            return Scene(self.size, parts, occupied)

    def add_tree(self):
        for _ in range(MAX_TRIES):
            x = 2 + np.random.randint(self.size[0] - 4)
            y = 0
            z = 2 + np.random.randint(self.size[2] - 4)
            tree = Tree.sample((x, y, z))
            positions = [block.pos for block in tree.blocks()]
            if any(self.occupied[pos] for pos in positions):
                continue
            parts = self._parts + (tree,)
            occupied = self.occupied.copy()
            for pos in positions:
                occupied[pos] = 1
            return Scene(self.size, parts, occupied)

    def add_block(self, block_type):
        for _ in range(MAX_TRIES):
            x = np.random.randint(self.size[0])
            y = 0
            z = np.random.randint(self.size[2])
            pos = (x, y, z)
            if self.occupied[pos]:
                continue
            block = Block(pos, block_type)
            parts = self._parts + (block,)
            occupied = self.occupied.copy()
            occupied[pos] = 1
            return Scene(self.size, parts, occupied)

    def blocks(self):
        for part in self._parts:
            if isinstance(part, Block):
                yield part
            else:
                for block in part.blocks():
                    yield block

    def parts(self):
        for part in self._parts:
            if isinstance(part, Block):
                yield part
                continue
            for subpart in part.parts():
                yield subpart

    def remove(self, rem):
        parts = []
        count = 0
        for part in self._parts:
            if part == rem:
                count += 1
            else:
                # TODO clean this up
                if isinstance(part, Block):
                    npart, ncount = part, 0
                else:
                    npart, ncount = part.remove(rem)
                count += ncount
                parts.append(npart)
        assert count == 1, 'removed %d copies of %s' % (count, rem)

        occupied = np.zeros(self.size)
        for part in parts:
            blocks = [part] if isinstance(part, Block) else part.blocks()
            for block in blocks:
                if occupied[block.pos] != 0:
                    raise BuilderException(
                        '%s space already occupied when removing %s' % (part, rem))
                occupied[block.pos] = 1

        return Scene(self.size, parts, occupied)

    def dump(self, label, fh):
        voxels = np.zeros(self.size, dtype=np.int32)
        for block in self.blocks():
            voxels[block.pos] = block.block_type.mat_id()
        data = {
            'voxels': voxels.tolist(),
            #'dimensions': self.size,
            'position': (0, 1, 0),
            'label': label
        }
        json.dump(data, fh)

