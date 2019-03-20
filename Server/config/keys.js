// keys.js - return development or production credentials
if(process.env.NODE_ENV === 'production'){ // production env
    module.exports = require('./prod'); // get the production credentials
} else { // development env
    module.exports = require('./dev'); // get the development credentials
}