const mongoose = require('mongoose');
const Song = mongoose.model('songs');
const User = mongoose.model('users');
const Playlist = mongoose.model('playlists')
const hash = require('object-hash');
module.exports = (app) => {
    let count = 0;
    let songsLength = 0;
    // app.get('/api/songCreate', (req, res) => {
    //     let songProt = {
    //         name:   "Song 13",
    //         artist: "Artist 13",
    //         album:  "Album 13",
    //     }
    //     let id = hash.MD5({name: songProt.name, artist: songProt.artist, album: songProt.album})
    //     let song = {
    //         songId: id,
    //         name:   "Song 13",
    //         artist: "Artist 13",
    //         album:  "Album 13",
    //         url:    "https://www.youtube.com/watch?v=0_hUI_fbKe8"
    //     }
    //     Song.findOne({songId: song.songId})
    //         .then((existingSong) => {
    //             if(existingSong){
    //                 let str = `The song with the name ${existingSong.name} already exists`;
    //                 res.send(str);
    //             }else{
    //                 new Song(song)
    //                 .save()
    //                 .then((song) => {
    //                     let string = 'Song created succesfully. here is the song';
    //                     res.send(string + JSON.stringify(song), null, 4);
    //                 })
    //                 .catch((err) => {
    //                     res.status(400).send(err);
    //                 })
    //             }
    //         })

    // });

    app.get('/api/songs', (req, res) => {
        
        let songProt = {
            name:   "Song 13",
            artist: "Artist 13",
            album:  "Album 13"
        }
        let id = hash.MD5({name: songProt.name, artist: songProt.artist, album: songProt.album})

        // songSkipped = count * 5;
        // if(songsLength < 5) count = 0;
        // Song.find()
        //     .skip(songSkipped)
        //     .limit(5)
        //     .exec((err, songs) => {
        //         songsLength = songs.length;
        //         if(songs.length < 5) count = 0;
        //         else count++;
        //         console.log(`song skipped: ${songSkipped} songsLength: ${songsLength}`)
        //         res.send(songs)
        //     })
        Song.find({songId: id})
            .then( songs => {
                res.send(songs);
            })
            .catch(err => {
                res.send(err);
            })
        })
            
    });

    app.get('/api/songDetails', (req,res) => {
        let id = hash.MD5({name: req.query.name, artist: req.query.artist, album: req.query.album})
        let id2 = hash.MD5({name: req.query.name, artist: req.query.artist, album: req.query.album})
        Song.findOne({songId: id})
            .then( song => {
                if(!song) song = {};
                res.send(song);
            })
            .catch( err => {
                res.send(err);
            })
    });

    /* PLAYLIST ROUTES */
    app.post('/api/createPlaylist', (req, res) => {
        let userName = req.query.userName;

        User.findOne({name: userName})
            .then( user => {
                if(!user){
                    res.status(404).send("User not found!");
                }else{
                    let playlistName = req.query.playlistName;
                    Playlist.findOne({name: playlistName})
                            .then( playlist => {
                                if(playlist){
                                    res.send("Playlist with the same name already exists!")
                                }else{
                                    new Playlist (
                                        {
                                            name: playlistName,
                                            songs: []
                                        }
                                    )
                                    .save()
                                    .then( playlist => {
                                        user.playlists.push(playlist);
                                        user.save();
                                        res.send(user);
                                    })
                                }
                            })
                }
            })
    });

    // app.get('/api/modifyPlaylist', (req,res) => {
    //     let userName = req.query.userName;
    //     let playlistName = req.query.playlistName;
    //     let id = hash.MD5({name: req.query.name, artist: req.query.artist, album: req.query.album})

    //     User.findOne({name: userName})
    //         .then( user => {
    //             if(!user){
    //                 res.status(404).send("User not found!");
    //             }else{
    //                 // res.send(user);
    //                 user.findOne({ playlists: {$elemMatch: {name: playlistName}}})
    //                     .then(playlist => {
    //                         if(!playlist) res.send("chacha")
    //                         res.send("asdasdasd");
    //                     })
    //                     .catch( err => {
    //                         res.send("err")
    //                     })
    //                 // let songid = hash.MD5({name: req.query.name, artist: req.query.artist, album: req.query.album})
    //                 // Song.findOne({songId: songid})
    //                 //     .then( song => {
    //                 //         if(song){
    //                 //             user.playlists.songs.push(song.id);
    //                 //             res.send(user);
    //                 //         }else{
    //                 //             res.send(user)
    //                 //         }
    //                 //     })
    //                 //     .catch( err => {
    //                 //         res.send(err);
    //                 //     })
    //             }
    //         })
    //         .catch( err => {
    //             res.send(error);
    //         })
    // })
}