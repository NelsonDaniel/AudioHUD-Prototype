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

// Called once.
function setup() {
  createCanvas(windowWidth, windowHeight)
  maxDiameter = 100; 
	theta = 0;
  x = 100
  y = 100
  // // Init values.
  // radius = windowHeight * 0.45
  // angle = 0
  // angle_vel = 0.01
  
  
}

// Called periodically.
function draw() { 
  clear()
  background('rgba(255, 255, 255, 0.1)')
  //background(0, 0, 0, 100);
  
  colorMode(HSB, 360, 100, 100, 100);
  noLoop();
  blendMode(ADD);
  var diam = 50 + sin(theta) * maxDiameter ;
  
  // draw the circle 
  for (let r = 0.0; r < 0.5; r += 0.01) {
    noFill();
    fill(0, 90, 5, 100);
    //circle(windowWidth * 0.4, windowHeight * 0.4, windowWidth * r);
    stroke('rgba(255, 255, 255, 0.1)')
    if(theta >= 3.14){
      noStroke();
      x += 90;
      y += 100;
      theta = 0;
    }
    ellipse(x, y, diam * r);
    
  }  
  loop();

  if(theta <= 3.14){
    theta += .03; 
  }  
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

function dispIcon(x, y){
  blendMode(ADD);
  for (let r = 0.0; r < 0.5; r += 0.01){
    stroke('red'); 
    strokeWeight(30);
    point(x, y);
  }
}

// Called every time the window is resized.
function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
  radius = windowHeight * 0.5
}