class SongsController {
    constructor($http, uiGridConstants, BACKEND_URL) {
        this.$http = $http;
        this.uiGridConstants = uiGridConstants;
        this.url = BACKEND_URL;

        this.songs = [];
    }

    getSongs() {
        return this.$http.get(this.url + 'api/songs/getSongs')
            .then(response => {
                return response.data;
            });
    }

    $onInit() {
/*         this.$http.get(this.url + 'api/current_user', {withCredentials: true}).then((response) => console.log(response.data ))*/

        console.log(this);

        this.getSongs().then((songs) => {
            console.log(songs)
            this.songs = songs.map(song => {

                delete song._id;
                delete song.songId;
                delete song.__v;

                return song;
            })
        this.gridOptions = {
            data: this.songs,
            enableRowSelection: true,
            enableFiltering: true,
            enableVerticalScrollbar: this.uiGridConstants.scrollbars.ALWAYS,
            enableHorizontalScrollbar: this.uiGridConstants.scrollbars.WHEN_NEEDED,
            columnDefs: [
                {
                    name: 'name',
                    filter: {
                        placeholder: 'Search...'
                    }
                },
                {
                    name: 'artist',
                    filter: {
                        placeholder: 'Search...'
                    }
                },
                {
                    name: 'album',
                    filter: {
                        placeholder: 'Search...'
                    }
                }
            ]
        }
        }
    )

        /* s */

        
    }
}

angular.module('poorify').component('songs', {
    controller: SongsController,
    template: require('./songs-template.html'),
    bindings: {
        user: '<'
    }
})