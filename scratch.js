const express = require('express');
const app = express();

app.get('/', (req, res) => {
  res.send('Not Found');
});

app.listen(3001, () => {
  console.log('Test server running');
});
