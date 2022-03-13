/* eslint-disable */
//const {ipcRenderer} = require('electron');

// Global variables.
//let x = 0.0;
//let y = 0.0;
//let c = 'NOTHING';

// Global variables
var angle;
var quadr; 
var sect; 
var mult; 
var mult2; 
var randImg;

let Computer_keyboard; 
let Cupboard_open_or_close; 
let Drawer_open_or_close; 
let Female_speech_and_woman_speaking; 
let Finger_snapping;
let Keys_jangling;
let Knock;
let Laughter; 
let Male_speech_and_man_speaking; 
let Printer;
let Scissors;
let Telephone;
let Writing;

// ipcRenderer.on('detection-to-display', (_, detection) => {
//   x = detection.x;
//   y = detection.y;
//   c = detection.c;
// });

function preload(){
  Computer_keyboard = loadImage('proto_icons/keyboard.png');
}

// Called once.
// eslint-disable-next-line no-unused-vars
function setup() {
  createCanvas(windowWidth, windowHeight);
}

// Called periodically.
// eslint-disable-next-line no-unused-vars
function draw() {
  clear();
  background('rgba(255, 255, 255, 0.1)');
  textSize(18);
  text('hello', windowWidth / 2, windowHeight / 2);
  image(Computer_keyboard, windowWidth/2, windowHeight/2);
}
