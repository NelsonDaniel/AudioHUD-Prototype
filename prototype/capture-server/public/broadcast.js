const peer = new Peer('peer-capture', {
  host: 'localhost',
  path: '/peer',
  port: 8000,
  debug: true,
});

window.peer = peer;

peer.on('open', async (id) => {
  peer.call('peer-model', window.stream);
});

const videoElement = document.querySelector('video');
async function getStream() {
  const stream = await navigator.mediaDevices.getDisplayMedia(
      {
        video: true,
        audio: {
          channelCount: 2,
          sampleRate: 32000,
        },
      },
  );

  window.stream = stream;
  videoElement.srcObject = stream;
  return stream;
};


const startBtn = document.getElementById('videoSelectBtn');
startBtn.onclick = (e) => {
  getStream()
  startBtn.classList.add('is-danger');
  startBtn.innerText = 'Recording';
};
