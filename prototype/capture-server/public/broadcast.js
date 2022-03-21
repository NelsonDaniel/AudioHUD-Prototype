const peer = new Peer('peer-capture', {
  host: 'localhost',
  path: '/peer',
  port: 8000,
  debug: true,
});

window.peer = peer;
peer.on('open', async (id) => {
  peer.call('peer-model', await getStream());
});

const videoElement = document.querySelector('video');
async function getStream() {
  const stream = await navigator.mediaDevices.getDisplayMedia(
      {
        video: true,
        audio: {
          noiseSuppression: true,
          channelCount: 2,
          sampleRate: 32000,
        },
      },
  );

  videoElement.srcObject = stream;
  return stream;
};
