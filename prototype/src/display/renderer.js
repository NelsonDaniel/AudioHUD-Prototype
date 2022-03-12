// Global variables.
let angle;
let angleVel;
let radius;

// Called once.
// eslint-disable-next-line no-unused-vars
function setup() {
  createCanvas(windowWidth, windowHeight);

  // Init values.
  radius = windowHeight * 0.45;
  angle = 0;
  angleVel = 0.01;
}

// Called periodically.
// eslint-disable-next-line no-unused-vars
function draw() {
  clear();
  background('rgba(255, 255, 255, 0.1)');

  // Translate the origin point to the center of the screen.
  translate(windowWidth / 2, windowHeight / 2);

  // Convert polar to cartesian.
  const x = radius * cos(angle);
  const y = radius * sin(angle);

  const v0 = createVector(0, 0);
  const v1 = createVector(x, y);
  drawArrow(v0, v1, 'yellowgreen');

  noStroke();

  // Update angle to move line.
  angle += angleVel;
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
  const arrowSize = 7;
  translate(vec.mag() - arrowSize, 0);
  triangle(0, arrowSize / 2, 0, -arrowSize / 2, arrowSize, 0);
  pop();
}

// Called every time the window is resized.
// eslint-disable-next-line no-unused-vars
function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
  radius = windowHeight * 0.5;
}

// const fs = require('fs');
// const path = require('path');

// // Global variables
// let angle;
// let quadr;
// let sect;
// let mult;
// let mult2;
// let randomIcon;
// let iconsNames = [];
// const icons = {};

// // eslint-disable-next-line no-unused-vars
// function preload() {
//   const iconsPath = path.join(__dirname, 'icons');
//   iconsNames = fs.readdirSync(path.join(__dirname, 'icons'));
//   iconsNames.forEach((n) => {
//     icons[n] = loadImage(path.join(iconsPath, n));
//   });
// }

// // Called once.
// // eslint-disable-next-line no-unused-vars
// function setup() {
//   createCanvas(windowWidth, windowHeight);
//   maxDiameter = 100;
//   theta = 0;

//   iconsz = 40; // ADJUST ICON SIZE
//   dvsn = 32; // ADJUST NUMBER OF SEGMENTS IN ARC

//   start = 0;
//   end = PI / (dvsn / 2);

//   noFill();
//   strokeWeight(5);
//   frameRate(1);
// }

// // Called periodically.
// // eslint-disable-next-line no-unused-vars
// function draw() {
//   clear();
//   noFill();
//   drawingContext.shadowBlur = 0;
//   background('rgba(255, 255, 255, 0.1)');
//   stroke('rgba(255, 255, 255, 0.1)');
//   circle(windowWidth / 2, windowHeight / 2, 100);
//   mult = Math.random() < 0.5 ? -1 : 1;
//   mult2 = Math.random() < 0.5 ? -1 : 1;
//   num = Math.random();
//   num2 = Math.random();
//   randomIcon = iconsNames[Math.floor(Math.random() * iconsNames.length)];
//   const sect = getSector(mult * num, mult2 * num2);
//   for (let i = 0; i < dvsn; i++) {
//     if (i == sect) {
//       setGlow();
//       image(
//           randomIcon,
//           iconX(sect) - iconsz / 2,
//           iconY(sect) - iconsz / 2,
//           iconsz,
//           iconsz,
//       );
//       arc(windowWidth / 2, windowHeight / 2, 600, 600, start, end);
//     } else if (sect == 4.5) {
//       setGlow();
//       image(
//           randomIcon,
//           windowWidth / 2 - iconsz / 2,
//           windowHeight / 2 - iconsz / 2,
//           iconsz,
//           iconsz,
//       );
//       circle(windowWidth / 2, windowHeight / 2, 100);
//     } else if (sect == 0.5) {
//       if (i == 0 || i == 31) {
//         setGlow();
//         image(randomIcon, iconX(31), iconY(31), iconsz, iconsz);
//         arc(windowWidth / 2, windowHeight / 2, 600, 600, start, end);
//       } else {
//         drawInactiveArc();
//       }
//     } else if (sect == 1.5) {
//       if (i == 23 || i == 24) {
//         setGlow();
//         image(randomIcon, iconX(23), iconY(23), iconsz, iconsz);
//         arc(windowWidth / 2, windowHeight / 2, 600, 600, start, end);
//       } else {
//         drawInactiveArc();
//       }
//     } else if (sect == 2.5) {
//       if (i == 15 || i == 16) {
//         setGlow();
//         image(randomIcon, iconX(16), iconY(16), iconsz, iconsz);
//         arc(windowWidth / 2, windowHeight / 2, 600, 600, start, end);
//       } else {
//         drawInactiveArc();
//       }
//     } else if (sect == 3.5) {
//       if (i == 7 || i == 8) {
//         setGlow();
//         image(randomIcon, iconX(8), iconY(8), iconsz, iconsz);
//         arc(windowWidth / 2, windowHeight / 2, 600, 600, start, end);
//       } else {
//         drawInactiveArc();
//       }
//     } else {
//       drawInactiveArc();
//     }
//     start += PI / (dvsn / 2);
//     end += PI / (dvsn / 2);
//   }
// }

// function drawInactiveArc() {
//   stroke('rgba(255, 255, 255, 0.1)');
//   drawingContext.shadowBlur = 0;
//   arc(windowWidth / 2, windowHeight / 2, 600, 600, start, end);
// }

// function setGlow() {
//   stroke(255);
//   drawingContext.shadowBlur = 20;
//   drawingContext.shadowColor = color(207, 7, 70);
// }

// function iconX(sector) {
//   sector = dvsn - sector;
//   const deg = 180 / dvsn + (360 / dvsn) * (sector - 1);
//   return windowWidth / 2 + 250 * Math.cos((deg * PI) / 180);
// }

// function iconY(sector) {
//   sector = dvsn - sector;
//   const deg = 180 / dvsn + (360 / dvsn) * (sector - 1);
//   return windowHeight / 2 - 250 * Math.sin((deg * PI) / 180);
// }

// function getSector(x, y) {
//   if (x > 0 && y == 0) {
//     return 0.5;
//   }
//   if (x == 0 && y > 0) {
//     return 1.5;
//   }
//   if (x < 0 && y == 0) {
//     return 2.5;
//   }
//   if (x == 0 && y < 0) {
//     return 3.5;
//   }
//   if (x == 0 && y == 0) {
//     return 4.5;
//   }
//   angle = Math.abs((180 / PI) * Math.atan(y / x));
//   quadr = getQuad(x, y);
//   if (quadr == 2) {
//     angle = 180 - angle;
//   }
//   if (quadr == 3) {
//     angle = 180 + angle;
//   }
//   if (quadr == 4) {
//     angle = 360 - angle;
//   }
//   sect = Math.floor(angle / (360 / dvsn));
//   sect = dvsn - sect;
//   return sect;
// }

// function getQuad(x, y) {
//   if (x > 0 && y > 0) {
//     return 1;
//   }
//   if (x < 0 && y > 0) {
//     return 2;
//   }
//   if (x < 0 && y < 0) {
//     return 3;
//   } else {
//     return 4;
//   }
// }

// // Called every time the window is resized.
// // eslint-disable-next-line no-unused-vars
// function windowResized() {
//   resizeCanvas(windowWidth, windowHeight);
//   radius = windowHeight * 0.5;
// }
