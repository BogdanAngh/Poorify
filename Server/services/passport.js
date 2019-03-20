const passport = require('passport');
const GoogleStrategy = require('passport-google-oauth20').Strategy;
const mongoose = require('mongoose');
const keys = require('../config/keys');

const User = mongoose.model('users');

passport.serializeUser((user, done) => {
    done(null, user.id)
});

passport.deserializeUser((id, done) => {
    User.findById(id)
        .then((user) => {
            done(null, user);
        })
});

//middleware that gives the app access to the google profile of the use
passport.use(
    new GoogleStrategy({
        clientID:       keys.googleClientID,
        clientSecret:   keys.googleClientSecret,
        callbackURL:    '/auth/google/callback',
        proxy: true
    }, 
    (accessToken, refreshToken, profile, done) => {
        User.findOne({ googleId: profile.id })
            .then((existingUser) => {
                if(existingUser){ // user is present in DB
                   done(null, existingUser); // tell passport that everything went fine (user exists)
                }else{ // create a new record in DB
                    new User( 
                        { 
                            googleId:   profile.id,
                            name:       profile.displayName 
                        } 
                    ).save()
                     .then((user) => done(null, user));  // tell passport that everything went fine (user has been saved)
                }
            })
    })  
);