#ifndef MAP_HPP
#define MAP_HPP
#include "objects.hpp"
#include <opencv2/opencv.hpp>

extern nlohmann::json settings;

class Map{
    public:
        /// Constructor.
        Map(){}
        /// Fields.
        cv::Mat image;
        State origin;
        long double resolution;
        int width; 
        int height;
        int dilation; 
        long double epsilon; 
        Segments segments;
        /// Methods.
        void find_segments(); 

};
#endif