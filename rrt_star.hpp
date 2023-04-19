#ifndef RRT_STAR_HPP
#define RRT_STAR_HPP
#include "objects.hpp"
#include <vector>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

class RRT_Star{
    public:
        RRT_Star(State& s_init, State& s_goal, cv::Mat& image, State& origin, long double& resolution, Segments& obstacles_segments){
            this->s_init = s_init;
            this->s_goal = s_goal;
            this->image = image;
            this->origin = origin;
            this->resolution = resolution;
            this->width = image.cols;
            this->height = image.rows;
            this->obstacles_segments = obstacles_segments;
        }
        RRT_Star(){}
        Path build(); /// Tree generation.
        void render(Path&); /// Visualization.
        void clean_up(); /// Deallocation.

    private:
        State s_init; /// Initial state.
        State s_goal; /// Goal state.
        int iterations; /// Number of states to generate.
        long double time_step; /// Time step.
        long double linear_velocity; /// Linear velocity.
        nlohmann::json angular_velocity; /// Angular velocity.
        long double search_radius; /// Search radius.
        long double goal_threshold; /// Goal threshold.
        long double max_steering_angle; /// Max Steering Angle.
        /// ---
        cv::Mat image; /// Image is an object that stores the matrix of the loaded image.
        State origin; /// Origin is the coordinate of the bottom-left corner that corresponds to the pixel location (0,0) of the matrix.
        long double resolution; /// Resolution refers to the size of each cell or pixel in the occupancy grid map represented by a PGM file, expressed in meters per pixel.
        int width; /// Width of the map in pixels.
        int height; /// Height of the map in pixels.
        Segments obstacles_segments; /// Collection of segments of polygons composing obstacles.
        // ---
        int state_count; /// State counter.
        State* state_array; /// State array.
        // ---
        void initialize(void); /// Initialize.
        void insert_root(void); /// Insert root.
        State sample_free_space(void); /// Sample free space.
        State* find_nearest(State&); /// Find nearest.
        State steer(State*, State&); /// Steer.
        bool is_obstacle_free(State&); /// Is obstacle free (Point).
        bool is_obstacle_free(State&, State&); /// Is obstacle free (Segment).
        std::vector<State*> get_neighbors(State&); /// Get neighbors.
        State* choose_parent(State&, std::vector<State*>&); /// Choose parent.
        State* insert(State&, State*); /// Insert.
        void rewire_tree(State*, std::vector<State*>&); /// Rewire tree.
        Path shortest_path(void); /// Shortest path.
        // ---
        long double calculate_euclidean_distance(long double&, long double&, long double&, long double&); /// Calculate euclidean disntace.
        long double calculate_distance(long double&); /// Calculate distance from linear velocity.
        long double calculate_angle(long double&); /// Caclulate distance from angular velocity.
        long double calculate_cost(State&, State&); /// Calculate cost.
        Velocity inverse_odometry(State&, State&); /// Calculate velocities.
        bool do_intersect(cv::Point&, cv::Point&, cv::Point&, cv::Point&); /// Verify the intersection of a segment with obstacles.
        int orientation(cv::Point&, cv::Point&, cv::Point&); /// Returns the orientation of a triplet.
        bool on_segment(cv::Point&, cv::Point&, cv::Point&); /// Returns the collinearity of a triplet.
        void swap(long double*, long double*); /// Swaps two elements of an array.
        int x_convert_to_pixel(long double&); /// Convert x-axis real coordinate to map coordinate
        int y_convert_to_pixel(long double&); /// Convert y-axis real coordinate to map coordinate
        long double get_random(long double&); /// Takes a negative double and return a random number between [-arg, arg].
};
#endif