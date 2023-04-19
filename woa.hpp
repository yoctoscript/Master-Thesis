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
            this->sInit = sInit;
            this->sGoal = sGoal;
            this->a = settings["WOA_a"];
            this->population = settings["WOA_population"];
            this->iterations = settings["WOA_iterations"];
        }
        void InitializePopulation();
        State CalculateFitness();
        void CircleUpdate(State&);
        void SpiralUpdate(State&);
        void RandomUpdate(State&);
        void CheckBoundary();
        State Optimize();

    private:
        State sInit;
        State sGoal;
        int population;
        int iterations;
        double explorationFactor;
        double spiralParameter;
        double probabilityParameter;
        double lowerBound;
        double upperBound;
        double a;
        State* whalesArray;

        double GenerateRandom(double a, double b);
        int GenerateRandom(int a, int b);
};


#endif