const passport = require('passport');

module.exports = (app) => {
    //when the user accesses '/auth/google', gets kicked into the OAuth flow, with the strategy google (Google OAauth) 
    app.get(
        '/auth/google', 
        passport.authenticate('google', {
            scope: ['profile', 'email'] //we want access to profile information and user email
        })
    );

    //after the user gives permission, it gets redirected to '/auth/google/callback' and passport handles the rest
    app.get(
        '/auth/google/callback',
        passport.authenticate('google')
    );
}