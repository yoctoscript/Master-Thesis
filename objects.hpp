#ifndef OBJECTS_HPP
#define OBJECTS_HPP
#include <vector>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

class State{
    public:
        State(State* p_val, double x_val, double y_val, double t_val, double v_val, double w_val, double c_val){
            p = p_val;
            x = x_val;
            y = y_val;
            t = t_val;
            v = v_val;
            w = w_val;
            c = c_val;
        }
        State(double x_val, double y_val){
            x = x_val;
            y = y_val;
        }
        State(){}
        // ---
        State* p; // Pointer to parent state.
        double x; // X-axis coordinate.
        double y; // Y-axis coordinate.
        double t; // Z-axis orientation.
        double v; // Linear velocity
        double w; // Angular velocity
        double c; // Cost
};

class Segment{
    public:
        Segment(cv::Point& p_val, cv::Point& q_val){
            p = p_val;
            q = q_val;
        }
        Segment(){}
        cv::Point p; /// First point.
        cv::Point q; /// Second point.
};

class Velocity{
    public:
        double v; /// Linear velocity.
        double w; /// Angular velocity.
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