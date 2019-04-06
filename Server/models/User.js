const mongoose = require('mongoose');
const { Schema } = mongoose; // get the mongoose.Schema obj
const Playlist   = mongoose.model('playlists')

const userSchema = new Schema({
    googleId: String,
    name: String,
    playlists: [ Playlist.schema ]
});

mongoose.model('users', userSchema);