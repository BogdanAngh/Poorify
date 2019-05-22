/**
 * @jest-environment node
 */

const functions = require('./routeFunctions')
const axios = require('axios')

describe('[GET] Users Tests: ', () => {
    it('Number of users should be equal to 3', () => {
        return functions.fetchUsers()
        .then(data => expect(data.length).toBe(3))
    })
    
    it('Name of first user should be Anghelache Bogdan', () => {
        return functions.fetchUsers()
        .then(data => expect(data[0].name).toEqual('Anghelache Bogdan'))
    })
    
    it('Name of second user should be Andrei Gorneanu', () => {
        return functions.fetchUsers()
        .then(data => expect(data[1].name).toEqual('Andrei Gorneanu'))
    })
    
    it('Name of second user should be Stefan Nita', () => {
        return functions.fetchUsers()
        .then(data => expect(data[2].name).toEqual('Stefan Nita'))
    })
})

describe('[GET] Playlists Tests (User: Anghelache Bogdan)', () =>{
    let user = 'Anghelache Bogdan'
    // it('Number of playlists should be 5', () => {
    //     return functions.fetchPlaylistsOfUser(user)
    //     .then(data => expect(data.length).toBe(5))
    // })

    it('Name of first playlist should be Rock', () => {
        return functions.fetchPlaylistsOfUser(user)
        .then(data => expect(data[0].name).toEqual('Rock'))
    })

    it('Name of second playlist should be Jazz', () => {
        return functions.fetchPlaylistsOfUser(user)
        .then(data => expect(data[1].name).toEqual('Jazz'))
    })

    it('Name of third playlist should be Blues', () => {
        return functions.fetchPlaylistsOfUser(user)
        .then(data => expect(data[2].name).toEqual('Blues'))
    })

    it('Name of forth playlist should be Country', () => {
        return functions.fetchPlaylistsOfUser(user)
        .then(data => expect(data[3].name).toEqual('Country'))
    })

    it('Blues playlist should have 2 songs', () => {
        return functions.fetchPlaylistsOfUser(user)
        .then(data => expect(data[2].songs.length).toBe(2))
    })
})

describe('[POST] Post requests to DB:', () => {
    let userName = 'Anghelache Bogdan'
    let playlistName = 'Manele'
    it('Create playlist '+ playlistName + ' for user '+ userName , () => {
        return functions.createPlaylist(userName, playlistName)
        .then(data => expect(data.name).toEqual(playlistName))
    })
})
