const {ipcRenderer} = require('electron');
const zmq = require('zeromq');
const path = require('path');

ipcRenderer.on('run-model', async () => {
  spawnModelProcess();
  await runClient();
});

async function runClient() {
  //  Socket to talk to server
  const sock = new zmq.Request();
  sock.connect('tcp://localhost:5555');

  while (1) {
    await sock.send('send detections');
    const [result] = await sock.receive();
    ipcRenderer.send('detection-from-model', result.toString());
  }
}

function spawnModelProcess() {
  console.log('spawning process');
  const modelProgramPath = path.join(__dirname, 'model.py');
  command = [
    'conda activate',
    'py37',
    '&& python',
    modelProgramPath,
  ].join(' ');

  const modelProcess = require('child_process').exec(command);

  modelProcess.stderr.on('data', (data) => {
    console.error(`stderr: ${data}`);
    console.log(`stderr: ${data}`);
  });

  modelProcess.on('close', (code) => {
    console.log(`child process exited with code ${code}`);
  });
};
