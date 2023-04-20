#ifndef OBJECTS_HPP
#define OBJECTS_HPP
#include <vector>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

class State{
    public:
        /// Constructors.
        State(State* p, long double x, long double y, long double z, long double v, long double w, long double t, long double c){
            this->p = p;
            this->x = x;
            this->y = y;
            this->z = z;
            this->v = v;
            this->w = w;
            this->t = t;
            this->c = c;
        }
        State(long double x, long double y){
            this->x = x;
            this->y = y;
        }
        State(){}
        /// Fields.
        State* p; /// Pointer to parent state.
        long double x; /// X-axis coordinate.
        long double y; /// Y-axis coordinate.
        long double z; /// Z-axis orientation.
        long double v; /// Linear velocity.
        long double w; /// Angular velocity.
        long double t; /// Duration.
        long double c; /// Cost.
};

class Segment{
    public:
        /// Constructors.
        Segment(cv::Point& p, cv::Point& q){
            this->p = p;
            this->q = q;
        }
        Segment(){}
        /// Fields.
        cv::Point p; /// First point.
        cv::Point q; /// Second point.
};

class Velocity{
    public:
        /// Fields.
        long double v; /// Linear velocity.
        long double w; /// Angular velocity.
};

class Path{
    public:
        /// Constructors.
        Path(){};
        /// Fields.
        int size;
        State* array;
};

typedef std::vector<Segment> Segments;
#endif