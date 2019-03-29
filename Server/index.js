require('./models/User');
require('./models/Song');

const express       = require('express');
const mongoose      = require('mongoose');
const authRoutes    = require('./routes/authRoutes');
const serviceRoutes = require('./routes/serviceRoutes');
const keys          = require('./config/keys');
const cookieSession = require('cookie-session');
const passport      = require('passport');
 

require('./services/passport');


mongoose.connect(keys.mongoURI);
 
const app = express();

app.set('json spaces', 2);

app.use(
    cookieSession({
        maxAge: 30 * 24 * 60 * 60 * 1000, // cookie lasts for 30 days 
        keys: [keys.cookieKey]
    })  
);
app.use(passport.initialize());
app.use(passport.session());

authRoutes(app);
serviceRoutes(app);

const PORT = process.env.PORT || 5000;
app.listen(PORT, "0.0.0.0", () => {
    console.log(`App listening on port ${ PORT }`);
})