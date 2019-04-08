var express = require('express')
var router = express.Router()

const mongoose = require('mongoose');
const Playlist = mongoose.model('playlists');
const User = mongoose.model('users');

/* PLAYLIST ROUTES */

// POST: adds a playlist (specified by name) to a certain user (specified by name)
// PARAMS: userName, playlistName
router.post('/createPlaylist', (req, res) => { //creates playlist
    let userName = req.query.userName;
    let duplicate = false;
    User.findOne({name: userName})
        .then( user => {
            if(!user){
                res.status(404).send({ error: "User not found!"});
            }else{
                let playlistName = req.query.playlistName;
                var playlists = user.playlists;

                for(let i = 0; i < playlists.length; i++){
                    if(playlistName == playlists[i].name){
                        duplicate = true;
                        res.send({ error: "Playlist with the same name already exists!" })
                        break;
                    }
                }

                if(!duplicate){
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

// router.get('/getPlaylists', (req, res) => {

//     Playlist.find({})
//             .then( playlists => {
//                 res.send(playlists);
//             })
// })

module.exports = router