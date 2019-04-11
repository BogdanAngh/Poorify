import * as angular from "angular";

angular.module('poorify').config(
    function ($stateProvider, $urlRouterProvider) {
        $stateProvider
            .state('login', {
                url: '/',
                component: 'login'
            })
            .state('songs', {
                url: '/songs',
                component: 'songs'
            });

        $urlRouterProvider.otherwise('/')
    }
)