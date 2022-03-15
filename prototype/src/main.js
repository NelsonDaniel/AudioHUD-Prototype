// Modules to control application life and create native browser window
const {app, BrowserWindow} = require('electron');
const path = require('path');
const zmq = require('zeromq');

let mainWindow;
let modelWindow;
const pids = [];

function createWindow() {
  // Create the browser window.
  mainWindow = new BrowserWindow({
    width: 800, maxWidth: 800, minWidth: 800,
    height: 600, maxHeight: 600, minHeight: 600,
    frame: false,
    autoHideMenuBar: true,
    transparent: true,
    useContentSize: true,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  });

  // set window to stay on top of any window
  mainWindow.setAlwaysOnTop(true, 'screen-saver');

  // and load the index.html of the app.
  mainWindow.loadFile(path.join(__dirname, 'index.html'));
  // Open the DevTools.
  // mainWindow.webContents.openDevTools();
}

app.on('ready', () => {
  createWindow();
  spawnModelProcess();
});

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  }

  if (modelWindow == null) {
    createWindow();
  }
});

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

function spawnModelProcess() {
  const modelProgramPath = path.join(__dirname, 'model/dist/model.exe');
  const modelProcess = require('child_process').spawn(modelProgramPath);
  pids.push(modelProcess.pid);

  modelProcess.stderr.on('data', (data) => {
    console.error(`stderr: ${data}`);
    console.log(`stderr: ${data}`);
  });

  modelProcess.on('close', (code) => {
    console.log(`child process exited with code ${code}`);
  });

  runClient();
};

//  Socket to talk to server
const sock = new zmq.Request();
async function runClient() {
  //  Socket to talk to server
  sock.connect('tcp://localhost:5555');
  sock.linger = 0;
  try {
    while (1) {
      await sock.send('send detections');
      const [detection] = await sock.receive();
      mainWindow.webContents.send('model-detection', detection.toString());
    }
  } finally {
    if (!sock.closed) {
      sock.close();
    }
  }
}

app.on('before-quit', function() {
  sock.disconnect('tcp://localhost:5555');
  sock.close();
  pids.forEach(function(pid) {
    // A simple pid lookup
    process.kill(pid, function( err ) {
      if (err) {
        throw new Error( err );
      } else {
        console.log( 'Process %s has been killed!', pid );
      }
    });
  });
});
