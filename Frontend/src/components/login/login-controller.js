import * as angular from "angular";

class LoginController {
    constructor($http) {
        this.$http = $http;
    }

    googleAuth() {
        console.log('ayyy')
        /* this.$http.get('https://secure-coast-35315.herokuapp.com/auth/google')
            .then(console.log) */
        this.$http({
            method: 'get',
            url: 'https://secure-coast-35315.herokuapp.com/auth/google',
        });
    }

    $onInit() {
        console.log('lmao');
    }
}

angular.module('poorify').component('login', {
    controller: LoginController,
    template: require('./login-template.html')
})