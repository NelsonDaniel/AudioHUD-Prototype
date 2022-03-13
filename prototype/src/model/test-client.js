const zmq = require('zeromq');

async function runClient() {
  console.log('Connecting to hello world server…');

  //  Socket to talk to server
  const sock = new zmq.Request();
  sock.connect('tcp://localhost:5555');

  while (1) {
    await sock.send('send detections');
    const [result] = await sock.receive();
    console.log('Received: ', result.toString());
  }
}

runClient();
