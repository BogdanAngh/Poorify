const mongoose     = require('mongoose');
const { Schema }   = mongoose;

const playlistSchema = new Schema({
    name:   String,
    songs: [{
        songId: String
    }]
});

mongoose.model('playlists', playlistSchema); 