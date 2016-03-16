var path = require("path");
var webpack = require("webpack");

module.exports = {
  entry: [
    "./js/app"
  ],
  module: {
    loaders: [
      {
        exclude: /node_modules/,
        loader: "babel",
        test: /\.js$/,
      }
    ]
  },
  output: {
    path: path.join(__dirname, "public/assets"),
    filename: "app.js",
    publicPath: "/assets/"
  }
};