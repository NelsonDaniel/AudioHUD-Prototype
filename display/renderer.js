// require("./p5") 

// Global variables.
let angle
let angle_vel
let radius

// Called once.
function setup() {
  createCanvas(windowWidth, windowHeight)
  
  // Init values.
  radius = windowHeight * 0.45
  angle = 0
  angle_vel = 0.01
}

// Called periodically.
function draw() {
  clear()
  background('rgba(255, 255, 255, 0.1)')

  // Translate the origin point to the center of the screen.
  translate(windowWidth / 2, windowHeight / 2)

  // Convert polar to cartesian.
  let x = radius * cos(angle)
  let y = radius * sin(angle)
  
  let v0 = createVector(0, 0);
  let v1 = createVector(x, y);
  drawArrow(v0, v1, 'yellowgreen')

  noStroke();

  // Update angle to move line.
  angle += angle_vel
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

// Called every time the window is resized.
function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
  radius = windowHeight * 0.5
}