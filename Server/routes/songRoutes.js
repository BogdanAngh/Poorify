var express = require('express')
var router = express.Router()

const mongoose = require('mongoose');
const Song = mongoose.model('songs');
const hash = require('object-hash');

 /* SONGS ROUTES */

 // POST: adds a song to the database
 // PARAMS: name, artist, album, genre, url
 router.post('/songCreate', (req, res) => {
    let songName    = req.query.name,
        songArtist  = req.query.artist,
        songAlbum   = req.query.album,
        songGenre   = req.query.genre, 
        songURL     = req.query.url 
        
    let songProt = {
        name:   songName,
        artist: songArtist,
        album:  songAlbum,
    }
    let id = hash.MD5({name: songProt.name, artist: songProt.artist, album: songProt.album})
    let song = {
        songId: id,
        name:   songName,
        artist: songArtist,
        album:  songAlbum,
        genre:  songGenre, 
        url:    songURL
    }
    Song.findOne({url: song.url})
        .then((existingSong) => {
            if(existingSong){
                let str = `The song with that URL already exists`;
                res.send(str);
            }else{
                new Song(song)
                .save()
                .then((song) => {
                    res.send(JSON.stringify(song), null, 4);
                })
                .catch((err) => {
                    res.status(400).send(err);
                })
            }
        })

});

// GET: returns all the songs in the database
router.get('/getSongs', (req, res) => { 
    Song.find({})
        .then( songs => {
            res.send(songs);
        })
        .catch(err => {
            res.send(err);
        })
})
        

// GET: returns the details of one song, specified by name, artist and album
// PARAMS: name, artist, album
router.get('/getSong', (req,res) => { 
    let id = hash.MD5({name: req.query.name, artist: req.query.artist, album: req.query.album})
    
    Song.findOne({songId: id})
        .then( song => {
            if(!song) song = {};
            res.send(song);
        })
        .catch( err => {
            res.send(err);
        })
});

// GET: returns the details of one song, specified by id
// PARAMS: songId
router.get('/getSongById', (req, res) => {
    let songId = req.query.songId;

    Song.findOne({ songId: songId})
        .then( song => {
            if(!song){
                res.status(404).send({ error: "Song not found!"})
            }
            res.send(song);
        })
        .catch(error => {
            res.status(500).send({ error: error})
        })
})

module.exports = router