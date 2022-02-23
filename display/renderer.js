// require("./p5") 

// Global variables.
let angle
let angle_vel
let radius

// Called once.
function setup() {
  createCanvas(windowWidth, windowHeight)
  
  // Init values.
  radius = height * 0.45
  angle = 0
  angle_vel = 0.1
}

// Called periodically.
function draw() {
  clear()
  background('rgba(0,255,0, 0.25)')

  // Translate the origin point to the center of the screen.
  translate(width / 2, height / 2)

  // Convert polar to cartesian.
  let x = radius * cos(angle)
  let y = radius * sin(angle)
  
  line(0, 0, x,y)

  // Update angle to move line.
  angle += angle_vel
}

// Called every time the window is resized.
function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
  radius = height * 0.45
}