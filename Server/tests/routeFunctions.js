const axios = require('axios')

const functions = {
    fetchUsers: () => 
        axios
        .get('http://localhost:5000/api/user/getUsers')
        .then(res => res.data)
        .catch(err => 'err' )
    ,
    fetchPlaylistsOfUser: (userName) =>
        axios
        .get('http://localhost:5000/api/user/getPlaylists?userName=' + userName)
        .then(res => res.data)
        .catch(err => 'err' )
    ,
    createPlaylist: (userName, playlistName) =>
        axios
        .post('http://localhost:5000/api/user/createPlaylist?userName='+ userName +'&playlistName=' + playlistName)
        .then(res => res.data)
        .catch(err => err)

}



module.exports = functions