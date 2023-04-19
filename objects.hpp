#ifndef OBJECTS_HPP
#define OBJECTS_HPP
#include <vector>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

class State{
    public:
        State(State* p, long double x, long double y, long double t, long double v, long double w, long double c){
            this->p = p;
            this->x = x;
            this->y = y;
            this->t = t;
            this->v = v;
            this->w = w;
            this->c = c;
        }
        State(long double x, long double y){
            this->x = x;
            this->y = y;
        }
        State(){}
        // ---
        State* p; // Pointer to parent state.
        long double x; // X-axis coordinate.
        long double y; // Y-axis coordinate.
        long double t; // Z-axis orientation.
        long double v; // Linear velocity
        long double w; // Angular velocity
        long double c; // Cost
};

class Segment{
    public:
        Segment(cv::Point& p, cv::Point& q){
            this->p = p;
            this->q = q;
        }
        Segment(){}
        cv::Point p; /// First point.
        cv::Point q; /// Second point.
};

class Velocity{
    public:
        long double v; /// Linear velocity.
        long double w; /// Angular velocity.
};

class Path{
    public:
        Path(int&, State&);
        Path(){};
        int size; /// Number of states.
        State* array; /// Pointer to the array.
};

typedef std::vector<Segment> Segments;
#endif