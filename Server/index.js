require('./models/Playlist')
require('./models/Song')
require('./models/User')


const express       = require('express');
const mongoose      = require('mongoose');
const authRoutes    = require('./routes/authRoutes');
const serviceRoutes = require('./routes/serviceRoutes');
const keys          = require('./config/keys');
const cookieSession = require('cookie-session');
const passport      = require('passport');
const cors          = require('cors') 
const bodyParser    = require('body-parser');
const app = express();
app.use(
    cookieSession({
        name: 'session',
        maxAge: 30 * 24 * 60 * 60 * 1000, // cookie lasts for 30 days 
        keys: [keys.cookieKey]
    })  
);

// app.use(function(req, res, next) {  
//     res.header('Access-Control-Allow-Origin', req.headers.origin);
//     res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
//     next();
// });

app.use(passport.initialize());
app.use(passport.session());


app.set('trust proxy', 1)



app.use(bodyParser.urlencoded());
app.use(bodyParser.json());

app.use(cors({
    'origin': 'http://localhost:8080',
    'credentials': true,
    'methods': 'GET, HEAD, PUT, PATCH, POST, DELETE',
    'preflightContinue': false
}));

app.set('json spaces', 2);





require('./services/passport');


mongoose.connect(keys.mongoURI);
 


authRoutes(app);
serviceRoutes(app);

const PORT = process.env.PORT || 5000;
app.listen(PORT, "0.0.0.0", () => {
    console.log(`App listening on port ${ PORT }`);
})