class NavbarController {
    constructor() {

    }


}

angular.module('poorify').component('navbar', {
    controller: NavbarController,
    template: require('./navbar-template.html'),
    transclude: true
})