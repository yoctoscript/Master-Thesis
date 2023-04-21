#ifndef RRT_STAR_HPP
#define RRT_STAR_HPP
#include "objects.hpp"
#include <vector>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

class RRT_Star{
    public:
        /// Constructors.
        RRT_Star(State& sInit, State& sGoal){
            this->sInit = sInit;
            this->sGoal = sGoal;
        }
        RRT_Star(){}
        /// Fields.
        State sInit; /// Initial state.
        State sGoal; /// Goal state.
        int iterations; /// Number of states to generate.
        long double timeStep; /// Time step.
        long double linearVelocity; /// Linear velocity.
        nlohmann::json angularVelocity; /// Angular velocity.
        long double searchRadius; /// Search radius.
        long double goalThreshold; /// Goal threshold.
        long double maxSteeringAngle; /// Max Steering Angle.
        int count; /// State counter.
        State* states; /// State array.
        /// Methods.
        Path Build(); /// Tree generation.
        void Initialize(); /// Initialize.
        void InsertRoot(); /// Insert root.
        State SampleFreeSpace(); /// Sample free space.
        State* FindNearest(State& sRand); /// Find nearest.
        State Steer(State* sNear, State& sRand); /// Steer.
        bool IsObstacleFree(State& sNew); /// Is obstacle free (Point).
        bool IsObstacleFree(State& sNew, State& sNeighbor); /// Is obstacle free (Segment).
        std::vector<State*> GetNeighbors(State& sNew); /// Get neighbors.
        State* ChooseParent(State& sNew, std::vector<State*>& neighbors); /// Choose parent.
        State* Insert(State& sNew, State* sParent); /// Insert.
        void RewireTree(State*, std::vector<State*>&); /// Rewire tree.
        Path ShortestPath(void); /// Shortest path.
        void Render(Path&); /// Visualization.
        void CleanUp(); /// Deallocation.
        /// Helper methods.
        long double CalculateEuclideanDistance(long double&, long double&, long double&, long double&); /// Calculate euclidean disntace.
        long double CalculateDistance(long double&); /// Calculate distance from linear velocity.
        long double CalculateAngle(long double&); /// Caclulate distance from angular velocity.
        long double CalculateCost(State&, State&); /// Calculate cost.
        Velocity InverseOdometry(State&, State&); /// Calculate velocities.
        bool DoIntersect(cv::Point&, cv::Point&, cv::Point&, cv::Point&); /// Verify the intersection of a segment with obstacles.
        int Orientation(cv::Point&, cv::Point&, cv::Point&); /// Returns the orientation of a triplet.
        bool OnSegment(cv::Point&, cv::Point&, cv::Point&); /// Returns the collinearity of a triplet.
        void Swap(long double*, long double*); /// Swaps two elements of an array.
        int XConvertToPixel(long double&); /// Convert x-axis real coordinate to map coordinate
        int YConvertToPixel(long double&); /// Convert y-axis real coordinate to map coordinate
        long double GenerateRandom(long double&); /// Takes a negative double and return a random number between [-arg, arg].
};
#endif