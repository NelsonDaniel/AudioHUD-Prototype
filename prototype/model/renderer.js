const {ipcRenderer, dialog} = require('electron');
const zmq = require('zeromq');
const path = require('path');

const peer = new Peer('peer-model', {
  host: 'localhost',
  path: '/peer',
  port: 8000,
  debug: true,
});

let mediaRecorder;
const recordedChunks = [];

const videoElement = document.querySelector('video');

peer.on('call', (call) => {
  console.log('answer!');
  call.answer();
  call.on('stream', (remoteStream) => {
    videoElement.srcObject = remoteStream;
    videoElement.play();

    // Create the Media Recorder
    const options = {mimeType: 'video/webm; codecs=vp9'};
    mediaRecorder = new MediaRecorder(remoteStream, options);

    // Register Event Handlers
    mediaRecorder.ondataavailable = handleDataAvailable;
    mediaRecorder.onstop = handleStop;
  });
});

const startBtn = document.getElementById('startBtn');
startBtn.onclick = (e) => {
  mediaRecorder.start();
  startBtn.classList.add('is-danger');
  startBtn.innerText = 'Recording';
};

const stopBtn = document.getElementById('stopBtn');
stopBtn.onclick = (e) => {
  mediaRecorder.stop();
  startBtn.classList.remove('is-danger');
  startBtn.innerText = 'Start';
};

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
  const modelScriptPath = path.join(__dirname, 'model.py');
  command = [
    'conda activate',
    'l3das',
    '&& python',
    modelScriptPath,
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

// Captures all recorded chunks
function handleDataAvailable(e) {
  console.log('video data available');
  recordedChunks.push(e.data);
}

// Saves the video file on stop
async function handleStop(e) {
  const blob = new Blob(recordedChunks, {
    type: 'video/webm; codecs=vp9',
  });

  const buffer = Buffer.from(await blob.arrayBuffer());

  const {filePath} = await dialog.showSaveDialog({
    buttonLabel: 'Save video',
    defaultPath: `vid-${Date.now()}.webm`,
  });

  if (filePath) {
    writeFile(filePath, buffer, () => console.log('video saved successfully!'));
  }
}
