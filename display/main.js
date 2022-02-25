const { app, BrowserWindow } = require('electron')

function createWindow () {
  const win = new BrowserWindow({
    width: 800, maxWidth: 800, minWidth: 800,
    height: 600, maxHeight: 600, minHeight: 600,
    frame: false,
    autoHideMenuBar: true,
    transparent: true,
    useContentSize: true
  })
  win.setAlwaysOnTop(true, 'screen-saver')
  win.loadFile('index.html')

  // remove comment for debugging
  // win.webContents.openDevTools()
}

app.whenReady().then(() => {
  createWindow()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow()
    }
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})
