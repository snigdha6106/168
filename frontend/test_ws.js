const WebSocket = require('ws');
const ws = new WebSocket('ws://127.0.0.1:8000/ws/telemetry');

ws.on('open', function open() {
  console.log('connected');
});

ws.on('message', function message(data) {
  console.log('received: %s', data.toString().substring(0, 100) + '...');
  ws.close();
});

ws.on('close', function close() {
  console.log('disconnected');
});

ws.on('error', function error(err) {
  console.log('error:', err);
});
