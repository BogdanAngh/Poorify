const mongoose = require('mongoose');
const Song = mongoose.model('songs');
const User = mongoose.model('users');
const Playlist = mongoose.model('playlists');

const userRoutes     = require('./userRoutes');
const songRoutes     = require('./songRoutes');
const playlistRoutes = require('./playlistRoutes')
module.exports = (app) => {
    let count = 0;
    let songsLength = 0;

    

    /* USER ROUTES */

    app.use('/api/user', userRoutes);
    
    /* SONGS ROUTES */

   app.use('/api/songs', songRoutes)

    /* PLAYLIST ROUTES */
    
    app.use('/api/playlists', playlistRoutes)

    
}