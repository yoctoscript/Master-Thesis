#include "woa.hpp"
#include <vector>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <random>

State WOA::Optimize()
{
    InitializePopulation();
    State sBest = CalculateFitness();
    int it = this->iterations;
    double a = this->a, r, A, C, l, p, b;
    while (it--)
    {
        for (int i = 0; i < this->population; ++i)
        {
            a -= (this->a / (this->iterations * this->population));
            r = GenerateRandom(0.0, 1.0);
            A = (2 * a * r) - a;
            C = (2 * r);
            l = GenerateRandom(-1.0, 1.0);
            p = GenerateRandom(0.0, 1.0);
            b = GenerateRandom(0.0, 1.0);
            if (p < 0.5)
            {
                /// Circle update
                if (fabs(A) < 1.0)
                {
                    double Dv = fabs((C * sBest.v) - whalesArray[i].v);
                    double Dw = fabs((C * sBest.w) - whalesArray[i].w);
                    whalesArray[i].v = sBest.v - (A * Dv);
                    whalesArray[i].w = sBest.w - (A * Dw);
                }
                /// Random update
                else
                {
                    int index = GenerateRandom(0, this->population);
                    State sRand = whalesArray[index];
                    double Dv = fabs((C * sRand.v) - whalesArray[i].v);
                    double Dw = fabs((C * sRand.w) - whalesArray[i].w);
                    whalesArray[i].v = sRand.v - (A * Dv);
                    whalesArray[i].w = sRand.w - (A * Dw);
                }
            }
            /// Spiral update
            else
            {
                double Dv = fabs(sBest.v - whalesArray[i].v);
                double Dw = fabs(sBest.w - whalesArray[i].w);
                whalesArray[i].v = Dv * exp(b*l) * cos(2* M_PI * l) + whalesArray[i].v;
                whalesArray[i].w = Dv * exp(b*l) * cos(2* M_PI * l) + whalesArray[i].w;
            }

        }
    }
}

double WOA::GenerateRandom(double a, double b)
{
    std::random_device rd;
    std::default_random_engine engine(rd());
    std::uniform_real_distribution<double> distribution(a, b);
    return distribution(engine);
}

int WOA::GenerateRandom(int a, int b)
{
    std::random_device rd;
    std::default_random_engine engine(rd());
    std::uniform_real_distribution<int> distribution(a, b);
    return distribution(engine);
}