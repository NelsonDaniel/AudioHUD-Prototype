const path = require('path');

const modelProgramPath = path.join(__dirname, 'model.py');
console.log(modelProgramPath);
command = [
  'conda run -n l3das22',
  '&& python',
  modelProgramPath].join(' ');

console.log(command);
require('child_process').exec(command);
