const mongoose = require('mongoose');
const Song = mongoose.model('songs');
const hash = require('object-hash');
module.exports = (app) => {
    app.get('/api/songs', (req, res) => {
        let songProt = {
            name:   "Song 1",
            artist: "Artist 1",
            album:  "Album 1",
            url:    "url 1"
        }
        let id = hash.MD5({name: songProt.name, artist: songProt.artist, album: songProt.album})
        let song = {
            songId: id,
            name:   "Song 1",
            artist: "Artist 1",
            album:  "Album 1",
        }
        Song.findOne({songId: song.songId})
            .then((existingSong) => {
                if(existingSong){
                    let str = `The song with the name ${existingSong.name} already exists`;
                    res.send(str);
                }else{
                    new Song(song)
                    .save()
                    .then((song) => {
                        let string = 'Song created succesfully. here is the song';
                        res.send(string + JSON.stringify(song), null, 4);
                    })
                    .catch((err) => {
                        res.status(400).send(err);
                    })
                }
            })

    });
}