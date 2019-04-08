var express = require('express')
var router = express.Router()

const mongoose = require('mongoose');
const User = mongoose.model('users');
const Playlist = mongoose.model('playlists');
const Song = mongoose.model('songs');
const hash = require('object-hash');

function isEmptyObject(obj) {
    return !Object.keys(obj).length;
}


/* USER ROUTES */

// GET: returns all the users in the database
router.get('/getUsers', (req, res) => {
    User.find({})
        .then( users => {
            res.send(users)
        })
})
// POST : creates a playlist for a certain user, specified by user name => 
// PARAMS: userName, playlistName
router.post('/createPlaylist', (req, res) => { 
    let userName = req.query.userName;
    let duplicate = false;

    User.findOne({name: userName})
        .then( user => {
            if(!user){
                res.status(404).send({ error: "User not found!"});
            }else{
                let playlistName = req.query.playlistName;
                var playlists = user.playlists;

//Can use / make a method findByName because the for loop is used in many places and this would make the code cleaner
                for(let i = 0; i < playlists.length; i++){
                    if(playlistName == playlists[i].name){
                        duplicate = true;
                        res.send({ error: "Playlist with the same name already exists!" })
                    }
                }

                if (!duplicate){
                    var playlist = new Playlist 
                        (
                            {
                                name: playlistName,
                                songs: []
                            }
                        )
                    user.playlists.push(playlist);
                    user.save();
                    res.send(playlist);
                }
            }
        })
});

// GET: gets the playlists of a user, specified by user name => 
// PARAMS: userName
router.get('/getPlaylists',(req, res) => {
    let userName = req.query.userName;

    User.findOne({name: userName})
        .then( user => {
            if(!user){
                res.status(404).send({ error: "User not found!" });
            }else{
                var playlists = user.playlists;
                res.send(playlists);
            }
        })
})

// GET: gets a certain playlist( specified by a playlist name) of a user, specified by user name => 
// PARAMS: userName, playlistName
router.get('/getPlaylistByName', (req, res) => {
    let userName     = req.query.userName
    var found = false;
    
        User.findOne({name: userName})
        .then( user => {
            if(!user){
                res.status(404).send({ error: "User not found!" });
            }else{
                let playlistName = req.query.playlistName;
                var playlists = user.playlists;
                
                for(let i = 0; i < playlists.length; i++){
                    if(playlistName == playlists[i].name){
                        res.send(playlists[i])
                        found = true; 
                        break;
                    }
                }
                if(!found){
                    res.send( { error: "Playlist not found" } )
                }
            }
        })
})

// POST: adds a song to the playlist of a user
// PARAMS: userName, playlistName, name, artist, album ( the last 3 are for the song to be added)
router.post('/addToPlaylist', (req,res) => {
    let userName = req.query.userName;
    let playlistName = req.query.playlistName;
    let id = hash.MD5({name: req.query.name, artist: req.query.artist, album: req.query.album})

    User.findOne({name: userName})
        .then( user => {
            if(!user){
                res.status(404).send({ error: "User not found!" });
            }else{
                var foundPlaylist = {};
                for (var i = 0; i < user.playlists.length; i++){
                    playlist = user.playlists[i]
                    if(playlist.name == playlistName){
                        foundPlaylist = playlist;
                        break;
                    }
                }
                // res.send(foundPlaylist)
                if(isEmptyObject(foundPlaylist)){
                    res.send({error: "Playlist not found"})
                }else{
                    Song.findOne({songId: id})
                        .then( song => {
                            if(song){
                                var songsInPlaylist = foundPlaylist.songs; 
                                for(var i = 0; i < songsInPlaylist.length; i++){
                                    if(song.songId == songsInPlaylist[i].songId){
                                        res.send({error: "Duplicate Song"})    
                                    }
                                }
                                foundPlaylist.songs.push({songId: song.songId})
                                foundPlaylist.save()
                                user.save()
                                res.send(foundPlaylist)
                            }else{
                                res.send({ error: "Song not found"})
                            }
                        })
                        .catch(err => {
                            res.send({error: err})
                        })
                }
            }
        })
        .catch( err => {
            res.send({error: err});
        })
})

module.exports = router
