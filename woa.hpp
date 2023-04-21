#ifndef WOA_HPP
#define WOA_HPP
#include "objects.hpp"
#include "logger.hpp"
#include <vector>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

extern nlohmann::json settings;

class WOA
{
    public:
        WOA(State sInit, State sGoal)
        {
            static Logger log(__FUNCTION__);
            this->sInit = sInit;
            this->sGoal = sGoal;
            this->_a = settings["WOA_a"];
            this->population = settings["WOA_population"];
            this->_iterations = settings["WOA_iterations"];
            this->linearVelocityMin = settings["WOA_linear_velocity_min"];
            this->linearVelocityMax = settings["WOA_linear_velocity_max"];
            this->angularVelocityMin = settings["WOA_angular_velocity_min"];
            this->angularVelocityMax = settings["WOA_angular_velocity_max"];
            this->a = this->_a;
            this->iterations = this->_iterations;
            log.trace("a: {:.2f} | population: {} | iterations {}", this->_a, this->population, this->iterations);
        }
        State Optimize();
        /// Fields.
        State sInit;
        State sGoal;
        State sBest;
        State sRand;
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
        int i;
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

class _WOA
{
    public:
        _WOA(){}
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


#endif