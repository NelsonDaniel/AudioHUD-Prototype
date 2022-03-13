// Modules to control application life and create native browser window
const {app, BrowserWindow, ipcMain} = require('electron');
const path = require('path');

let mainWindow;
let modelWindow;

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
  // mainWindow.webContents.openDevTools()
}

function createModelWindow() {
  // hidden worker
  modelWindow = new BrowserWindow({
    show: false,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  });

  modelWindow.loadFile(path.join(__dirname, 'model/index.html'));

  modelWindow.on('closed', () => {
    modelWindow = null;
  });

  modelWindow.on('ready-to-show', () => {
    modelWindow.webContents.send('run-model');
  });

  console.log('model worker created');
}

app.on('ready', () => {
  createWindow();
  createModelWindow();
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

ipcMain.on('model-detection', (_, detection) => {
  mainWindow.webContents.send('detection-to-display', detection);
});


