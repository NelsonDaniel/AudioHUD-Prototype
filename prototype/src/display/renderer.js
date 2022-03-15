/* eslint-disable */
const {ipcRenderer} = require('electron');

// Global variables
let x = 0.0;
let y = 0.0;
let c = 'NOTHING';

let angle;
let quadr;
let sect;

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
let Chink_and_clink;

ipcRenderer.on('model-detection', (_, detection) => {
  parsed = JSON.parse(detection.toString());
  c = parsed[0];
  x = parsed[1];
  y = parsed[2];
});

// eslint-disable-next-line no-unused-vars
function preload() {
  Computer_keyboard = loadImage('protoicons/keyboard.png');
  Cupboard_open_or_close = loadImage('protoicons/door-open.png');
  Drawer_open_or_close = loadImage('protoicons/drawer.png');
  Female_speech_and_woman_speaking = loadImage('protoicons/femchat.png');
  Finger_snapping = loadImage('protoicons/snap.png');
  Keys_jangling = loadImage('protoicons/key-chain.png');
  Knock = loadImage('protoicons/knock.png');
  Laughter = loadImage('protoicons/laugh.png');
  Male_speech_and_man_speaking = loadImage('protoicons/malechat.png');
  Printer = loadImage('protoicons/printer.png');
  Scissors = loadImage('protoicons/scissors.png');
  Telephone = loadImage('protoicons/phone.png');
  Writing = loadImage('protoicons/writing.png');
  Chink_and_clink = loadImage('protoicons/broken-glass.png')
}

// Called once.
// eslint-disable-next-line no-unused-vars
function setup() {
  createCanvas(windowWidth, windowHeight);
  iconsz = 40; // ADJUST ICON SIZE
  dvsn = 32; // ADJUST NUMBER OF SEGMENTS IN ARC

  start = 0;
  end = PI/(dvsn/2);

  noFill();
  strokeWeight(5);
}

// Called periodically.
// eslint-disable-next-line no-unused-vars
function draw() {
  clear();
  noFill();
  drawingContext.shadowBlur = 0;
  background('rgba(255, 255, 255, 0.05)');
  stroke('rgba(255, 255, 255, 0.05)');
  circle(windowWidth/2, windowHeight/2, 100);
  let sect = getSector(x, y);
  if(c == 'NOTHING'){
    sect = -1;
  }
  for(let i = 0; i < dvsn; i++) {
    if(i == sect) {
      setGlow();
      image(eval(c), iconX(sect) - (iconsz/2), iconY(sect) - (iconsz/2), iconsz, iconsz);
      arc(windowWidth / 2, windowHeight / 2, 600, 600, start, end);
    }else if(sect == 4.5){
      setGlow();
      image(eval(c), windowWidth/2 - (iconsz/2), windowHeight/2 - (iconsz/2), iconsz, iconsz);
      circle(windowWidth/2, windowHeight/2, 100);
    }else if(sect == 0.5){
      if(i == 0 || i == 31){
        setGlow();
        image(eval(c), iconX(31), iconY(31), iconsz, iconsz);
        arc(windowWidth / 2, windowHeight / 2, 600, 600, start, end);
      }else{ drawInactiveArc(); }
    }else if(sect == 1.5){
      if(i == 23 || i == 24){
        setGlow();
        image(eval(c), iconX(23), iconY(23), iconsz, iconsz);
        arc(windowWidth / 2, windowHeight / 2, 600, 600, start, end);
      }else{ drawInactiveArc(); }
    }else if(sect == 2.5){
      if(i == 15 || i == 16){
        setGlow();
        image(eval(c), iconX(16), iconY(16), iconsz, iconsz);
        arc(windowWidth / 2, windowHeight / 2, 600, 600, start, end);
      }else{ drawInactiveArc(); }
    }else if(sect == 3.5){
      if(i == 7 || i == 8){
        setGlow();
        image(eval(c), iconX(8), iconY(8), iconsz, iconsz);
        arc(windowWidth / 2, windowHeight / 2, 600, 600, start, end);
      }else{ drawInactiveArc(); }
    }else{
      drawInactiveArc();
    }
    start += PI/(dvsn/2); 
    end += PI/(dvsn/2);
  }
  // uncomment for debugging coordinates
  // printInfo(); 
}

function drawInactiveArc(){
  stroke('rgba(255, 255, 255, 0.1)');
  drawingContext.shadowBlur = 0;
  arc(windowWidth / 2, windowHeight / 2, 600, 600, start, end);
}

function setGlow(){
  stroke(255);
  drawingContext.shadowBlur = 50;
  drawingContext.shadowColor = color(207, 7, 70);
}

function iconX(sector){
  sector = dvsn - sector; 
  let deg = (180/dvsn) + ((360/dvsn) * (sector - 1));
  return (windowWidth/2 + (250 * Math.cos(deg * PI/180)));
}

function iconY(sector){
  sector = dvsn - sector;
  let deg = (180/dvsn) + ((360/dvsn) * (sector - 1));
  return (windowHeight/2 - (250 * Math.sin(deg * PI/180)));
 }

function getSector(x, y){
  if(x > 0 && y == 0){
    return 0.5;
  }
  if(x == 0 && y > 0){
    return 1.5;
  }
  if(x < 0 && y == 0){
    return 2.5;
  }
  if(x == 0 && y < 0){
    return 3.5;
  }
  if(x == 0 && y == 0){
    return 4.5;
  }
  angle = Math.abs((180/PI) * Math.atan(y/x));
  quadr = getQuad(x, y); 
  if(quadr == 2){
    angle = 180 - angle; 
  }
  if(quadr == 3){
    angle = 180 + angle; 
  }
  if(quadr == 4){
    angle = 360 - angle; 
  }
  sect = angle/(360/dvsn);
  sect = (sect < 1) ? Math.ceil(sect) : Math.floor(sect); 
  sect = (dvsn - sect);
  return sect;
}

function getQuad(x, y){
  if(x > 0 && y > 0){
    return 1;
  }
  if(x < 0 && y > 0){
    return 2; 
  }
  if(x < 0 && y < 0){
    return 3;
  }
  else{
    return 4; 
  }
}

function printInfo(){
  strokeWeight(1);
  stroke(1);
  textSize(18);
  text([c, x, y], windowWidth / 2, windowHeight / 2);
  strokeWeight(5);
}

// Called every time the window is resized.
function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
  radius = windowHeight * 0.5
}
