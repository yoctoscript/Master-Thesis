#include "rrt_star.hpp"
#include "objects.hpp"
#include "logger.hpp"
#include <fstream>
#include <iostream>
#include <inttypes.h>
#include <cstdlib>
#include <cmath>
#include <random>
#include <algorithm>
#include <limits>
#include <cstdlib>
#include <time.h>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

extern nlohmann::json settings;
extern Map map;

Path RRT_Star::build(){
    static Logger log(__FUNCTION__);
    log.debug("initialize()");
    initialize();
    log.debug("insert_root()");
    insert_root();
    for (state_count = 1; state_count < iterations; ++state_count){
        log.debug("State {}: sample_free_space()", state_count);
        State s_rand = sample_free_space();
        log.debug("State {}: find_nearest()", state_count);
        State* s_near = find_nearest(s_rand);
        log.debug("State {}: steer()", state_count);
        State s_new = steer(s_near, s_rand);
        if (is_obstacle_free(s_new)){
            log.debug("State {}: get_neighbors()", state_count);
            std::vector<State*> neighbors = get_neighbors(s_new);
            if (neighbors.empty()){
                log.warn("No valid neighbors");
                --state_count;
                continue;
            }
            log.debug("State {}: choose_parent", state_count);
            State* s_parent = choose_parent(s_new, neighbors);
            log.debug("State {}: insert()", state_count);
            State* s_new_ = insert(s_new, s_parent);
            log.debug("State {}: rewire_tree()", state_count);
            rewire_tree(s_new_, neighbors);
        }
        else{
            --state_count;
            continue;
        }
    }
    log.debug("shortest_path()");
    return shortest_path();
}

void RRT_Star::initialize(){
    int _iterations = settings["RRT*_iterations"];
    long double _time_step = settings["RRT*_time_step"];
    long double _linear_velocity = settings["RRT*_linear_velocity"];
    nlohmann::json _angular_velocity = settings["RRT*_angular_velocity"];
    long double _search_radius = settings["RRT*_search_radius"];
    long double _max_steering_angle = settings["RRT*_max_steering_angle"];
    long double _goal_threshold = settings["RRT*_goal_threshold"];
    iterations = _iterations;
    time_step = _time_step;
    linear_velocity = _linear_velocity;
    angular_velocity = _angular_velocity;
    search_radius = _search_radius;
    max_steering_angle = _max_steering_angle;
    goal_threshold = _goal_threshold;
    state_array = new State[_iterations];
    return;
}

void RRT_Star::insert_root(){
    state_array[0] = s_init;
    return;
}

State RRT_Star::sample_free_space(){
    static Logger log(__FUNCTION__);
    State s_rand;
    s_rand.x = get_random(origin.x);
    s_rand.y = get_random(origin.y);
    log.trace("s_rand ({:+.2f}, {:+.2f})", s_rand.x, s_rand.y);
    return s_rand;
}

State* RRT_Star::find_nearest(State& s_rand){
    static Logger log(__FUNCTION__);
    State *s_near;
    double shortest_distance = std::numeric_limits<double>::max();
    double distance;
    for(int i = 0; i < state_count; ++i) {
        distance = calculate_euclidean_distance(s_rand.x, s_rand.y, state_array[i].x, state_array[i].y);
        if (distance < shortest_distance){
            shortest_distance = distance;
            s_near = &state_array[i];
        }
    }
    log.trace("s_near ({:.2f}, {:.2f})", s_near->x, s_near->y);
    return s_near;
}

State RRT_Star::steer(State* s_near, State& s_rand){
    static Logger log(__FUNCTION__);
    std::vector<long double> angular_velocities = angular_velocity.get<std::vector<long double>>();
    int N = angular_velocities.size();
    long double angular_velocity[N], distance[N], x_temp[N], y_temp[N];
    // Copy angular velocities into the array.
    for (int i = 0; i < N; i++) {
        angular_velocity[i] = angular_velocities[i];
    }
    // Calculate resulting states and their distance to 's_rand'.
    for (int i = 0; i < N; ++i){
        x_temp[i] = (s_near->x) - calculate_distance(linear_velocity)*sin(s_near->t + calculate_angle(angular_velocity[i])/2.0);
        y_temp[i] = (s_near->y) + calculate_distance(linear_velocity)*cos(s_near->t + calculate_angle(angular_velocity[i])/2.0);
        distance[i] = calculate_euclidean_distance(s_rand.x, s_rand.y, x_temp[i], y_temp[i]);
    }
    // Sort arrays by distance.
    for (int i = 0; i < N-1; ++i)
        for (int j = 0; j < N-i-1; ++j)
            if (distance[j] > distance[j+1]){
                swap(&angular_velocity[j], &angular_velocity[j+1]);
                swap(&distance[j], &distance[j+1]);
                swap(&x_temp[j], &x_temp[j+1]);
                swap(&y_temp[j], &y_temp[j+1]);
            }
    // Choose the closest as 's_new'
    State s_new;
    s_new.x = x_temp[0];
    s_new.y = y_temp[0];
    s_new.t =  s_near->t + calculate_angle(angular_velocity[0]);
    log.trace("s_new ({:+.2f}, {:+.2f})", s_new.x, s_new.y);
    return s_new;
}

bool RRT_Star::is_obstacle_free(State& s_new){
    static Logger log("is_obstacle_free[Point]");
    int x = x_convert_to_pixel(s_new.x);
    int y = y_convert_to_pixel(s_new.y);
    cv::Point p(x, y);
    for (auto& segment: obstacles_segments){
        if (do_intersect(p, p, segment.p, segment.q)){
            log.warn("State ({:.2f}, {:.2f}) is not free", s_new.x, s_new.y);
            return false;
        }
    }
    log.trace("State ({:+.2f}, {:+.2f}) is free", s_new.x, s_new.y);
    return true;
}

bool RRT_Star::is_obstacle_free(State& s_new, State& s_neighbor){
    static Logger log("is_obstacle_free[Segment]");
    int p_x = x_convert_to_pixel(s_new.x);
    int p_y = y_convert_to_pixel(s_new.y);
    cv::Point p(p_x, p_y);
    int q_x = x_convert_to_pixel(s_neighbor.x);
    int q_y = y_convert_to_pixel(s_neighbor.y);
    cv::Point q(q_x, q_y);
    for (auto& segment: obstacles_segments){
        if (do_intersect(p, q, segment.p, segment.q)){
            log.warn("Segment p({:+.2f}, {:+.2f}) q({:+.2f}, {:+.2f}) is not free", s_new.x, s_new.y, s_neighbor.x, s_neighbor.y);
            return false;
        }
    }
    log.trace("Segment p({:+.2f}, {:+.2f}) q({:+.2f}, {:+.2f}) is free", s_new.x, s_new.y, s_neighbor.x, s_neighbor.y);
    return true;
}

std::vector<State*> RRT_Star::get_neighbors(State& s_new){
    std::vector<State*> neighbors;
    for (int i = 0; i < state_count; ++i){
        double distance = calculate_euclidean_distance(s_new.x, s_new.y, state_array[i].x, state_array[i].y);
        if (distance < search_radius){
            if (fabs(s_new.t - state_array[i].t) < max_steering_angle){
                if (is_obstacle_free(s_new, state_array[i])){
                    neighbors.push_back(&state_array[i]);
                }
            }
        }
    }
    return neighbors;
}

State* RRT_Star::choose_parent(State& s_new, std::vector<State*>& neighbors){
    State* s_parent;
    double lowest_cost = std::numeric_limits<double>::max();
    for (auto& neighbor: neighbors){
        double cost = neighbor->c + calculate_cost(s_new, *neighbor);
        if (cost < lowest_cost){
            lowest_cost = cost;
            s_parent = neighbor;
        }
    }
    return s_parent;
}

State* RRT_Star::insert(State& s_new, State* s_parent){
    Velocity vel = inverse_odometry(s_new, *s_parent);
    double cost = calculate_cost(s_new, *s_parent);
    s_new.p = s_parent;
    s_new.v = vel.v;
    s_new.w = vel.w;
    s_new.c = s_parent->c + cost;
    state_array[state_count] = s_new;
    return &state_array[state_count];
}

void RRT_Star::rewire_tree(State* s_new, std::vector<State*>& neighbors){
    for (auto& neighbor: neighbors){
        double cost = s_new->c + calculate_cost(*s_new, *neighbor);
        if (cost < neighbor->c){
            Velocity vel = inverse_odometry(*neighbor, *s_new);
            neighbor->p = s_new;
            neighbor->c = cost;
            neighbor->v = vel.v;
            neighbor->w = vel.w;
        }
    }
    return;
}

void RRT_Star::render(Path& path){
    cv::Mat output(image.size(), CV_8UC3);
    cv::cvtColor(image, output, cv::COLOR_GRAY2BGR);
    std::vector<cv::Mat> channels(3);
    cv::split(output, channels);
    channels[0] = image.clone();
    channels[1] = image.clone();
    channels[2] = image.clone();
    cv::merge(channels, output);
    int x, y;
    /// Draw all states.
    for (int i = 1; i < state_count; ++i){
        State a = state_array[i];
        State b = *(a.p);
        x = x_convert_to_pixel(a.x);
        y = y_convert_to_pixel(a.y);
        cv::Point p(x, y); 
        x = x_convert_to_pixel(b.x);
        y = y_convert_to_pixel(b.y);
        cv::Point q(x, y); 
        line(output, p, q, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }
    /// Initial configuration.
    x = x_convert_to_pixel(s_init.x);
    y = y_convert_to_pixel(s_init.y);
    cv::Point init(x, y);
    circle(output, init, 5, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
    /// Goal threshold region and configuration.
    x = x_convert_to_pixel(s_goal.x);
    y = y_convert_to_pixel(s_goal.y);
    cv::Point goal(x, y);
    int goal_radius = (int)(goal_threshold / resolution);
    circle(output, goal, goal_radius, cv::Scalar(157, 157, 255), -1, cv::LINE_AA);
    circle(output, goal, 5, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
    /// Draw shortest path.
    for (int i = 1; i < path.size; ++i){
        State a = path.array[i];
        State b = *(a.p);
        x = x_convert_to_pixel(a.x);
        y = y_convert_to_pixel(a.y);
        cv::Point p(x, y); 
        x = x_convert_to_pixel(b.x);
        y = y_convert_to_pixel(b.y);
        cv::Point q(x, y); 
        line(output, p, q, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
    }
    /// Draw obstacle segments.
    for (auto& segment: obstacles_segments){
        line(output, segment.p, segment.q, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    }
    cv::imshow("RRT*", output);
}

Path RRT_Star::shortest_path(void){
    static Logger log(__FUNCTION__);
    double lowest_cost = std::numeric_limits<double>::max();
    State* best_state = nullptr;
    /// Find the state that lies within goal region and has the lowest cost.
    for (int i = 0; i < state_count; ++i){
        double distance_from_goal = calculate_euclidean_distance(state_array[i].x, state_array[i].y, s_goal.x, s_goal.y);
        if (distance_from_goal < goal_threshold){
            if (state_array[i].c < lowest_cost){
                lowest_cost = state_array[i].c;
                best_state = &state_array[i];
            }
        }
    }
    /// If no path found, return.
    Path path;
    if (!best_state){
        path.size = 0;
        path.array = nullptr;
        return path;
    }
    log.trace("Best State: (x={:+.2f}, y={:+.2f}, t={:+.2f}, c={:+.2f})",best_state->x, best_state->y, best_state->t, best_state->c);
    /// Find how many state constitute the shortest path.
    int i = 1;
    State* pointer = best_state;
    while (pointer->p){
        ++i;
        pointer = pointer->p;
    }
    /// Copy the states into the path array.
    pointer = best_state;
    path.size = i;
    path.array = new State[i];
    for (int j = i-1; j >= 0; --j){
        path.array[j] = *pointer;
        pointer = pointer->p;
    }
    /// Log the complete path.
    for (int i = 0; i < path.size; ++i){
        log.trace("State {}: (x={:+.2f}, y={:+.2f}, t={:+.2f}, v={:+.2f}, w={:+.2f}, c={:+.2f})", i, path.array[i].x, path.array[i].y, path.array[i].t, path.array[i].v, path.array[i].w, path.array[i].c);
    }
    return path;
}

void RRT_Star::clean_up(){
    delete state_array;
}




























//**********************************************************************
// This is utility section
//**********************************************************************        




























Velocity RRT_Star::inverse_odometry(State& s_new, State& s_old){
    double delta_x = s_new.x - s_old.x;
    double delta_y = s_new.y - s_old.y;
    double delta_t = s_new.t - s_old.t;
    double distance = sqrt(pow(delta_x, 2)+pow(delta_y, 2));
    double linear_velocity = distance / time_step;
    double angular_velocity = delta_t / time_step;
    Velocity vel;
    vel.v = linear_velocity;
    vel.w = angular_velocity;
    return vel;
}

long double RRT_Star::calculate_euclidean_distance(long double& x, long double& y, long double& a, long double& b){
    return sqrt(pow(x-a, 2)+pow(y-b, 2));
}

long double RRT_Star::calculate_distance(long double& v){
    return (v * time_step);
}

long double RRT_Star::calculate_angle(long double& w){
    return (w * time_step);
}

void RRT_Star::swap(long double *a, long double *b){
    double temp = *a;
    *a = *b;
    *b = temp;
    return;
}

long double RRT_Star::calculate_cost(State& a, State& b){
    return calculate_euclidean_distance(a.x, a.y, b.x, b.y);
}

int RRT_Star::orientation(cv::Point& p, cv::Point& q, cv::Point& r){
    int val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
    if (val == 0) return 0;  // collinear
    return (val > 0)? 1: 2; // clock or counterclock wise
}

bool RRT_Star::on_segment(cv::Point& p, cv::Point& q, cv::Point& r){
    if (q.x <= std::max(p.x, r.x) && q.x >= std::min(p.x, r.x) && q.y <= std::max(p.y, r.y) && q.y >= std::min(p.y, r.y))
       return true;
    return false;
}

int RRT_Star::x_convert_to_pixel(long double& x){
    return (int)((x + (-origin.x)) / resolution);
}

int RRT_Star::y_convert_to_pixel(long double& y){
    return (int)((y + (-origin.y)) / resolution);
}

bool RRT_Star::do_intersect(cv::Point& p1, cv::Point& q1, cv::Point& p2, cv::Point& q2){
    int o1 = orientation(p1, q1, p2);
    int o2 = orientation(p1, q1, q2);
    int o3 = orientation(p2, q2, p1);
    int o4 = orientation(p2, q2, q1);
    if (o1 != o2 && o3 != o4)
        return true;
    if (o1 == 0 && on_segment(p1, p2, q1)) return true;
    if (o2 == 0 && on_segment(p1, q2, q1)) return true;
    if (o3 == 0 && on_segment(p2, p1, q2)) return true;  
    if (o4 == 0 && on_segment(p2, q1, q2)) return true;
    return false;
}

long double RRT_Star::get_random(long double& a){
    long double lower_bound = a;
    long double upper_bound = -a;
    std::default_random_engine re{std::random_device{}()};
    std::uniform_real_distribution<long double> unif(lower_bound, upper_bound);
    return unif(re);
}






