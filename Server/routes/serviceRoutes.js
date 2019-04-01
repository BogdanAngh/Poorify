const mongoose = require('mongoose');
const Song = mongoose.model('songs');
const hash = require('object-hash');
module.exports = (app) => {
    let count = 0;
    let songsLength = 0;
    app.get('/api/songCreate', (req, res) => {
        let songProt = {
            name:   "Song 8",
            artist: "Artist 8",
            album:  "Album 8",
            url:    "url 8"
        }
        let id = hash.MD5({name: songProt.name, artist: songProt.artist, album: songProt.album})
        let song = {
            songId: id,
            name:   "Song 8",
            artist: "Artist 8",
            album:  "Album 8",
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

    app.get('/api/songs', (req, res) => {
        
        songSkipped = count * 5;
        // if(songsLength < 5) count = 0;
        Song.find({})
            .skip(songSkipped)
            .limit(5)
            .exec((err, songs) => {
                songsLength = songs.length;
                if(songs.length < 5) count = 0;
                else count++;
                console.log(`song skipped: ${songSkipped} songsLength: ${songsLength}`)
                res.send(songs)
            })
            
    });
}