#ifndef MAP_HPP
#define MAP_HPP
#include "objects.hpp"
#include <opencv2/opencv.hpp>

extern nlohmann::json settings;

class Map{
    public:
        Map(cv::Mat& image, State& _origin, double& _resolution){
            this->image = image;
            this->origin = origin;
            this->resolution = resolution;
            this->width = image.cols;
            this->height = image.rows;
            this->dilation = settings["MAP_obstacles_dilation"];
            this->epsilon = settings["MAP_epsilon"]; 
        }
        Map(){} /// Default class constructor.
        Segments find_segments(); /// Returns the segments of polygons that make up obstacles.
    private:
        cv::Mat image; /// Image is an object that stores the matrix of the loaded image.
        State origin; /// Origin is the coordinate of the bottom-left corner that corresponds to the pixel location (0,0) of the matrix.
        double resolution; /// Resolution refers to the size of each cell or pixel in the occupancy grid map represented by a PGM file, expressed in meters per pixel.
        int width; /// Width of the map in pixels.
        int height; /// Height of the map in pixels.
        int dilation; /// Dilation is the number of pixels added to the boundary of an obstacle in an occupancy grid map during the dilation process.
        double epsilon; /// Epsilon is a parameter that specifies the maximum distance between the original curve and its approximation, where smaller epsilon values result in more precise approximations.
};
#endif