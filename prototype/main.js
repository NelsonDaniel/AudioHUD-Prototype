// Modules to control application life and create native browser window
const {app, BrowserWindow, ipcMain, shell} = require('electron');
const path = require('path');
const SERVER_PORT = 8000;

function createOverlayWindow() {
  // Create the browser window.
  const overlayWindow = new BrowserWindow({
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
  overlayWindow.setAlwaysOnTop(true, 'screen-saver');

  // and load the index.html of the app.
  overlayWindow.loadFile(path.join(__dirname, 'overlay', 'index.html'));
  // Open the DevTools.
  // mainWindow.webContents.openDevTools();

  overlayWindow.once('ready-to-show', () => {
    overlayWindow.webContents.send('run-model');
  });

  ipcMain.on('detection-from-model', (_, detection) => {
    overlayWindow.webContents.send('detection-for-display', detection);
  });
}

function createCaptureWindow() {
  // Create the browser window.
  const captureWindow = new BrowserWindow({
    // show: false,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  });

  // and load the index.html of the app.
  captureWindow.loadFile(path.join(__dirname, 'capture-server', 'server.html'));

  captureWindow.on('ready-to-show', () => {
    shell.openExternal(`http://localhost:${SERVER_PORT}/broadcast.html`);
  });

  // Open the DevTools.
  captureWindow.webContents.openDevTools();
}

function createModelWindow() {
  // hidden worker
  const modelWindow = new BrowserWindow({
    // show: false,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  });

  modelWindow.loadFile(path.join(__dirname, 'model', 'index.html'));

  modelWindow.once('ready-to-show', () => {
    modelWindow.webContents.send('run-model');
  });
}

app.whenReady().then(() => {
  createOverlayWindow();
  createCaptureWindow();
  createModelWindow();
  app.on('activate', function() {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});


// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

