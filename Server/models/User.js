const mongoose = require('mongoose');
const { Schema } = mongoose; // get the mongoose.Schema obj

const userSchema = new Schema({
    googleId: String,
    name: String
});

mongoose.model('users', userSchema);