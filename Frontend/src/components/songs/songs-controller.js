class SongsController {
    constructor($http, uiGridConstants) {
        this.$http = $http;
        this.uiGridConstants = uiGridConstants;
    }

    getSongs() {
        this.$http.get('https://secure-coast-35315.herokuapp.com/api/songs')
            .then(response => {
                this.songs = response.data;
            });
    }

    showSongs() {
        console.log(this.songs);
    }

    $onInit() {
        // this.getSongs();

        this.songs = [
            {
                name: "Song 13",
                artist: "Trinity Force (feat. Teo)",
                album: "Poverty"
            },
            {
                name: "Nothing Else Sandman",
                artist: "Megadeth",
                album: "Raining Blood"
            },
            {
                name: "Snowblind",
                artist: "Black Sabbath",
                album: "Vol. 4"
            },
            {
                name: "Vara nu dorm",
                artist: "Smiley",
                album: "Delia"
            }
        ];

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
}

angular.module('poorify').component('songs', {
    controller: SongsController,
    template: require('./songs-template.html')
})