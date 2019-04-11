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
        passport.authenticate('google'),
        (req, res) => {
            res.redirect('/#!/songs')
        }
    );
    
    // routes for logout
    app.get('/api/logout', (req, res) => {
        req.logout();
        var s = `You have logged out successfully.`;
        res.send(s);
    });

    app.get(
        '/api/current_user',
        (req, res) => {
            // console.log(req)
            res
            .send(req.user);
            // var headers = new Headers();
            //     headers.append('Content-Type', 'application/json');
            //     headers.append('Accept', 'application/json');

            // return fetch('/your/server_endpoint', {
            //     method: 'POST',
            //     mode: 'same-origin',
            //     redirect: 'follow',
            //     credentials: 'include', // Don't forget to specify this if you need cookies
            //     headers: headers,
            //     body: JSON.stringify({
            //         first_name: 'John',
            //         last_name: 'Doe'
            //     })
})



    app.get(
        '/',
        (req, res) => {
            res.send('HOME PAGE')
        }
    );
}