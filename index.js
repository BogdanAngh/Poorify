const express = require('express');
const app = express();  

app.get('/', (req, res) => {
    res.send({hi: 'there'});
});

const PORT = process.env.PORT || 5000; // process.env.PORT = environment var for production usage (Heroku); 5000 for dev usage

app.listen(5000);