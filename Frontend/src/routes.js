import * as angular from "angular";

angular.module('poorify').config(
    function ($stateProvider, $urlRouterProvider, $locationProvider) {
        $stateProvider
            .state('login', {
                url: '/',
                component: 'login'
            })
            .state('songs', {
                url: '/songs',
                component: 'songs',
                resolve: {
                    user: function($http) {
                        return $http.get('http://localhost:5000/api/current_user', {withCredentials: true}).then((response) => response.data)
                    }
                }
            });
            // $locationProvider.html5Mode({
            //     enabled: true,
            //     requireBase: false
            //   });
        $urlRouterProvider.otherwise('/')

        
    }
)