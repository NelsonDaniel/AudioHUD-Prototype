const express = require('express');
const app = express();
const server = require('http').createServer(app);
const port = 8000;

// eslint-disable-next-line new-cap
const {ExpressPeerServer} = require('peer');


// eslint-disable-next-line new-cap
const peerServer = ExpressPeerServer(server, {
  path: '/peer',
});

app.use(peerServer);
app.use(express.static(__dirname + '/public'));
app.get('/', (request, response) => {
  response.sendFile(__dirname + '/broadcast.html');
});

peerServer.on('connection', (conn) => {
  console.log(conn.getId());
});

server.listen(port, () => console.log(`Server is running on port ${port}`));
