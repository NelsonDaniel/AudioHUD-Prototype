const {ipcRenderer} = require('electron');

// Global variables.
let x = 0.0;
let y = 0.0;
let c = 'NOTHING';

ipcRenderer.on('detection-to-display', (_, detection) => {
  x = detection.x;
  y = detection.y;
  c = detection.c;
});

// Called once.
// eslint-disable-next-line no-unused-vars
function setup() {
  createCanvas(windowWidth, windowHeight);
}

// Called periodically.
// eslint-disable-next-line no-unused-vars
function draw() {
  clear();
  textSize(18);
  text([c, x, y], windowWidth / 2, windowHeight / 2);
}
