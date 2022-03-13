const {ipcRenderer} = require('electron');
const {classes} = require('./classes');

ipcRenderer.on('run-model', () => {
  const randCoord = () => {
    const max = 2;
    const min = -2;
    return Math.random() * (max - min) + min;
  };

  setInterval(() => {
    const c = classes[Math.floor(Math.random()*classes.length)];
    const x = randCoord();
    const y = randCoord();
    const z = randCoord();
    const detection = {c, x, y, z};
    ipcRenderer.send('model-detection', detection);
  }, 1000);
});


