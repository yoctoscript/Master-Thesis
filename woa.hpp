#ifndef WOA_HPP
#define WOA_HPP
#include "objects.hpp"
#include "logger.hpp"
#include <vector>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

extern nlohmann::json mySettings;

class WOA
{
    public:
        WOA(){}
        std::vector<State> optimizedPath;
        void Apply(Path path);
        bool TestCollision(State& a, State& b);
        bool TestAngle(State& a, State& b);
        int Orientation(cv::Point& p, cv::Point& q, cv::Point& r);
        bool OnSegment(cv::Point& p, cv::Point& q, cv::Point& r);
        int XConvertToPixel(long double& x);
        int YConvertToPixel(long double& y);
        bool DoIntersect(cv::Point& p1, cv::Point& q1, cv::Point& p2, cv::Point& q2);
};

class MinimalWOA
{
    public:
        MinimalWOA(State sInit, State sGoal)
        {
            static Logger log(__FUNCTION__);
            this->sInit = sInit;
            this->sGoal = sGoal;
            this->_a = mySettings["WOA_a"];
            this->a = this->_a;
            this->population = mySettings["WOA_population"];
            this->_iterations = mySettings["WOA_iterations"];
            this->iterations = this->_iterations;
            this->linearVelocityMin = mySettings["WOA_linear_velocity_min"];
            this->linearVelocityMax = mySettings["WOA_linear_velocity_max"];
            this->angularVelocityMin = mySettings["WOA_angular_velocity_min"];
            this->angularVelocityMax = mySettings["WOA_angular_velocity_max"];
            log.trace("a: {:.2f} | population: {} | iterations: {}", this->_a, this->population, this->iterations);
        }
        State Optimize(); /// Main function.
        /// Fields.
        State sInit; /// Initial state.
        State sGoal; /// Goal state.
        State sBest; /// Best state.
        State sRand; /// Random state.
        int i;
        int population;
        int _iterations; /// Constant
        int iterations;
        long double _a; /// Constant
        long double a;
        long double r;
        long double A;
        long double C;
        long double p;
        long double l;
        long double b;
        long double linearVelocityMin;
        long double linearVelocityMax;
        long double angularVelocityMin;
        long double angularVelocityMax;
        State* whales; /// Array of particles.
        /// Methods.
        void InitializePopulation();
        void CalculateFitness();
        void CoefficientUpdate();
        void CircleUpdate();
        void SpiralUpdate();
        void RandomUpdate();
        void CheckBoundary();
        long double GenerateRandom(long double a, long double b);
        int GenerateRandom(int a, int b);
        long double CalculateEuclideanDistance(long double& x, long double& y, long double& a, long double& b);
        long double CalculateDistance(long double& v);
        long double CalculateAngle(long double& w);
        void RenderParticles();
        int XConvertToPixel(long double& x);
        int YConvertToPixel(long double& y);
        void CleanUp();
};
#endif