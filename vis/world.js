var createEngine = require('voxel-engine');
var createTerrain = require('voxel-perlin-terrain');
var $ = require('jquery');

// create the game
var game = createEngine({
  chunkDistance: 2,
  materials: ['grass', 'redwool', 'plank', 'bluewool'],
  texturePath: './textures/',
  generate: function(x, y, z) {
    return y == 0 ? 1 : 0
  },
  worldOrigin: [0, 0, 0],
  controls: { discreteFire: true }
});
var container = document.getElementById('container');
game.appendTo(container);

var createPlayer = require('voxel-player')(game);
var shama = createPlayer('shama.png');
shama.yaw.position.set(0, 10, 0);
shama.possess();

$.getJSON("scene.json", load);

// add some trees
/*var createTree = require('voxel-forest');
for (var i = 0; i < 20; i++) {
  createTree(game, { bark: 4, leaves: 3 });
}*/

//var loaded = false;

game.on('tick', function(dt) {
  //if (!loaded) {
  //  loaded = true;
  //  $.getJSON("scene.json", load);
  //}
});

function load(data) {
  var voxels = data['voxels']
  var origin = data['position']
  console.log(data)
  console.log(voxels)
  console.log(voxels.length)
  for (var x = 0; x < voxels.length; x++) {
    for (var y = 0; y < voxels[x].length; y++) {
      for (var z = 0; z < voxels[x][y].length; z++) {
        var pos = [
          origin[0] + x,
          origin[1] + y,
          origin[2] + z
        ]
        game.setBlock(pos, voxels[x][y][z]);
      }
    }
  }
}
