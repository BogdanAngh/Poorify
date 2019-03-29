const mongoose     = require('mongoose');
const { Schema }   = mongoose;

const songSchema = new Schema({
    songId: String,
    name:   String,
    artist: String,
    album:  String,
    url:    String
});

mongoose.model('songs', songSchema);