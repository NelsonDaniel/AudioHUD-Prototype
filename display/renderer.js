// require("./p5") 

// Global variables.
// let angle
// let angle_vel
// let radius
var maxDiameter;
var theta;
var zero = false;
var x; 
var y;
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

function preload() {
  Computer_keyboard = loadImage('icons/keyboard.png');
  Cupboard_open_or_close = loadImage('icons/door-open.png');
  Drawer_open_or_close = loadImage('icons/drawer.png');
  Female_speech_and_woman_speaking = loadImage('icons/femchat.png');
  Finger_snapping = loadImage('icons/snap.png');
  Keys_jangling = loadImage('icons/key-chain.png');
  Knock = loadImage('icons/knock.png');
  Laughter = loadImage('icons/laugh.png');
  Male_speech_and_man_speaking = loadImage('icons/malechat.png');
  Printer = loadImage('icons/printer.png');
  Scissors = loadImage('icons/scissors.png');
  Telephone = loadImage('icons/phone.png');
  Writing = loadImage('icons/writing.png');
}

// Called once.
function setup() {
  createCanvas(windowWidth, windowHeight)
  maxDiameter = 100; 
	theta = 0;
  //x = 100
  //y = 100
  iconsz = 40; //adjust icon size
  dvsn = 32; //number of segments in arc
  start = 0;
  end = PI/(dvsn/2);
  imgArr = ["Computer_keyboard", "Cupboard_open_or_close", "Drawer_open_or_close", "Female_speech_and_woman_speaking", 
    "Finger_snapping", "Keys_jangling", "Knock", "Laughter", "Male_speech_and_man_speaking", "Printer", "Scissors", "Telephone",
    "Writing"];
  // // Init values.
  // radius = windowHeight * 0.45
  // angle = 0
  // angle_vel = 0.01
  noFill();  
  strokeWeight(5);
  frameRate(1);
}

// Called periodically.
function draw() {
  clear() 
  noFill();
  drawingContext.shadowBlur = 0;
  background('rgba(255, 255, 255, 0.1)')
  mult = Math.random() < 0.5 ? -1 : 1;
  mult2 = Math.random() < 0.5 ? -1 : 1;
  num = Math.random();
  num2 = Math.random();
  randImg = imgArr[Math.floor(Math.random() * 13)];
  let sect = getSector(mult * num, mult2 * num2);
  for(let i = 0; i < dvsn; i++){
    if(i == sect){
      image(eval(randImg), iconX(sect) - (iconsz/2), iconY(sect) - (iconsz/2), iconsz, iconsz);
      stroke(255);
      drawingContext.shadowBlur = 20;
      drawingContext.shadowColor = color(207, 7, 70);
    }else{
      stroke('rgba(255, 255, 255, 0.1)');
      drawingContext.shadowBlur = 0;
    }
    arc(windowWidth / 2, windowHeight / 2, 600, 600, start, end);
    start += PI/(dvsn/2); 
    end += PI/(dvsn/2);
  }
  
  // //background(0, 0, 0, 100);
  // colorMode(HSB, 360, 100, 100, 100);
  // noLoop();
  // blendMode(ADD);
  // var diam = 50 + sin(theta) * maxDiameter ;
  
  // for (let r = 0.0; r < 0.5; r += 0.01) {
  //   noFill();
  //   fill(0, 90, 5, 100);
  //   stroke('rgba(255, 255, 255, 0.1)')
  //   if(theta >= 3.14){
  //     noStroke();
  //     x += 90;
  //     y += 100;
  //     theta = 0;
  //   }
  //   ellipse(x, y, diam * r);
  // }  
  // loop();
  // if(theta <= 3.14){
  //   theta += .03; 
  // }  
}

// ref: https://p5js.org/reference/#/p5.Vector/magSq
function drawArrow(base, vec, myColor) {
  push();
  stroke(myColor);
  strokeWeight(5);
  fill(myColor);
  translate(base.x, base.y);
  line(0, 0, vec.x, vec.y);
  rotate(vec.heading());
  let arrowSize = 7;
  translate(vec.mag() - arrowSize, 0);
  triangle(0, arrowSize / 2, 0, -arrowSize / 2, arrowSize, 0);
  pop();
}

//for understanding coord system
function dispIcon(x, y){
  blendMode(ADD);
  for (let r = 0.0; r < 0.5; r += 0.01){
    stroke('red'); 
    strokeWeight(30);
    point(x, y);
  }
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
  sect = Math.floor(angle/(360/dvsn));
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

// Called every time the window is resized.
function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
  radius = windowHeight * 0.5
}