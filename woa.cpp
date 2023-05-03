#include "woa.hpp"
#include "map.hpp"
#include "logger.hpp"
#include "objects.hpp"
#include <vector>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <random>
#include "debug.hpp"

extern nlohmann::json mySettings;
extern Map myMap;
extern cv::VideoWriter* myVideo;

Path* WOA::Apply()
{
    #ifdef DEBUG
        static Logger log(__FUNCTION__);
        log.debug("InitializePopulation()");
    #endif
    InitializePopulation();
    #ifdef DEBUG
        log.debug("CalculateFitness()");
    #endif
    CalculateFitness();
                RenderParticles();

    while (this->iterations--)
    {
        for (this->i = 0; this->i < this->population; ++(this->i))
        {
            #ifdef DEBUG
                log.debug("CoefficientUpdate()");
            #endif
            CoefficientUpdate();
            if ((this->p) < 0.5l)
            //if (false)
            {
                if (fabs(this->A) < 0.5l)
                //if(true)
                {
                    #ifdef DEBUG
                        log.debug("CircleUpdate()");
                    #endif
                    CircleUpdate();
                }
                else
                {
                    #ifdef DEBUG
                        log.debug("RandomUpdate()");
                    #endif
                    RandomUpdate();
                }
            }
            else
            {
                #ifdef DEBUG
                    log.debug("SpiralUpdate()");
                #endif
                SpiralUpdate();
            }
        }
        #ifdef DEBUG
            log.debug("CheckBoundary");
        #endif
        CheckBoundary();
        #ifdef DEBUG
            log.debug("CalculateFitness");
        #endif
        CalculateFitness();
        #ifdef DEBUG
            log.debug("RenderParticles()");
        #endif
        if (this->iterations == 1)
            RenderParticles();
    }
    #ifdef DEBUG
        log.debug("CleanUp()");
    #endif
    CleanUp();
    return this->best;
}

void WOA::InitializePopulation()
{
    #ifdef DEBUG
        static Logger log(__FUNCTION__);
    #endif
    this->particles = new Path[this->population];
    particles[0].size = this->size;
    particles[0].array = new State[this->size];
    for (int i = 0; i < this->size; ++i)
    {
        particles[0].array[i] = rrtStarPath->array[i];
    }
    for (int i = 1; i < this->population; ++i)
    {
        particles[i].size = this->size;
        particles[i].array = new State[this->size];
        for (int j = 0; j < this->size; ++j)
        {
            if (j == 0)
            {
                this->particles[i].array[j].x = this->rrtStarPath->array[j].x;
                this->particles[i].array[j].y = this->rrtStarPath->array[j].y;
                continue;
            }
            else if (j == this->size-1)
            {
                this->particles[i].array[j].x = this->rrtStarPath->array[j].x;
                this->particles[i].array[j].y = this->rrtStarPath->array[j].y;
                continue;
            }
            this->particles[i].array[j].x = GenerateRandom(this->rrtStarPath->array[j].x - this->widthBound, this->rrtStarPath->array[j].x + this->widthBound);
            this->particles[i].array[j].y = GenerateRandom(this->rrtStarPath->array[j].y - this->heightBound, this->rrtStarPath->array[j].y + this->heightBound);
        }
    }
    this->greatest.size = this->size;
    this->greatest.array = new State[this->size];
    this->greatestFitness = std::numeric_limits<long double>::max();
}

void WOA::CalculateFitness()
{
    static Logger log(__FUNCTION__);
    #ifdef DEBUG
        static Logger log(__FUNCTION__);
    #endif
    int bestIndex = 0;
    this->bestFitness = std::numeric_limits<long double>::max();
    long double fitness, shortness, collision;
    for (int i = 0; i < this->population; ++i)
    {
        shortness = 0.0l;
        collision = 0.0l;
        for (int j = 1; j < this->size; ++j)
        {
            shortness += CalculateEuclideanDistance(this->particles[i].array[j-1].x, this->particles[i].array[j-1].y, this->particles[i].array[j].x, this->particles[i].array[j].y);
            collision += IsObstacleFree(this->particles[i].array[j-1], this->particles[i].array[j]) ? 0.0l : 99999.0l;
        }
        fitness = pathShortnessWeight * shortness + collisionFreeWeight * collision;
        if (fitness < this->bestFitness)
        {
            this->bestFitness = fitness;
            this->shortestLength = shortness;
            bestIndex = i;
        }

    }
    this->best = &this->particles[bestIndex];
    if (this->bestFitness < this->greatestFitness)
    {
        SaveGreatest();
        this->greatestFitness = this->bestFitness;
    }
    else
    {
        LoadGreatest();
    }
    #ifdef DEBUG
        log.trace("Best Whale: {}, Fitness: {:.2f}", bestIndex, bestFitness);
    #endif
}

void WOA::CoefficientUpdate()
{
    #ifdef DEBUG
        static Logger log(__FUNCTION__);
    #endif
    this->a -= (this->_a / (long double)(this->_iterations * this->population));
    this->r = GenerateRandom(0.0l, 1.0l);
    this->A = (2 * this->a * this->r) - this->a;
    this->C = (2 * this->r);
    this->l = GenerateRandom(-1.0l, 1.0l);
    this->p = GenerateRandom(0.0l, 1.0l);
    #ifdef DEBUG
        log.trace("a: {:.3f} | r: {:.2f} | A: {:.2f} | C: {:.2f} | l: {:.2f} | p: {:.2f} | b: {:.2f}", this->a, this->r, this->A, this->C, this->l, this->p, this->b);
    #endif
}

void WOA::CircleUpdate()
{
    for (int j = 1; j < this->size-1; ++j)
    {
        long double Dx = fabs((this->C * this->best->array[j].x) - this->particles[this->i].array[j].x);
        long double Dy = fabs((this->C * this->best->array[j].y) - this->particles[this->i].array[j].y);
        this->particles[this->i].array[j].x = this->best->array[j].x - (this->A * Dx);
        this->particles[this->i].array[j].y = this->best->array[j].y - (this->A * Dy);
    }
}

void WOA::RandomUpdate()
{
    int index = GenerateRandom(0, this->population);
    this->random = &this->particles[index];
     for (int j = 1; j < this->size-1; ++j)
    {
        long double Dx = fabs((this->C * this->random->array[j].x) - this->particles[this->i].array[j].x);
        long double Dy = fabs((this->C * this->random->array[j].y) - this->particles[this->i].array[j].y);
        this->particles[this->i].array[j].x = this->random->array[j].x - (this->A * Dx);
        this->particles[this->i].array[j].y = this->random->array[j].y - (this->A * Dy);
    }
}

void WOA::SpiralUpdate()
{
    for (int j = 1; j < this->size-1; ++j)
    {
        long double Dx = fabs(this->best->array[j].x - this->particles[this->i].array[j].x);
        long double Dy = fabs(this->best->array[j].y - this->particles[this->i].array[j].y);
        this->particles[this->i].array[j].x = Dx * exp(b*l) * cos(2* M_PI * l) + this->particles[this->i].array[j].x;
        this->particles[this->i].array[j].y = Dy * exp(b*l) * cos(2* M_PI * l) + this->particles[this->i].array[j].y;
    }
}

// void WOA::CheckBoundary()
// {
//     static Logger log(__FUNCTION__);
//     for (int i = 0; i < this->population; ++i)
//     {
//         for (int j = 1; j < this->size-1; ++j)
//         {
//             if (this->particles[i].array[j].x > (this->rrtStarPath->array[j].x + this->widthBound))
//             {
//                 this->particles[i].array[j].x = (this->rrtStarPath->array[j].x + this->widthBound);
//             }
//             else if (this->particles[i].array[j].x < (this->rrtStarPath->array[j].x - this->widthBound))
//             {
//                 this->particles[i].array[j].x = (this->rrtStarPath->array[j].x - this->widthBound);
//             }
//             if (this->particles[i].array[j].y > (this->rrtStarPath->array[j].y + this->heightBound))
//             {
//                 this->particles[i].array[j].y = (this->rrtStarPath->array[j].y + this->heightBound);
//             }
//             else if (this->particles[i].array[j].y < (this->rrtStarPath->array[j].y - this->heightBound))
//             {
//                 this->particles[i].array[j].y = (this->rrtStarPath->array[j].y - this->heightBound);
//             }
//         }
        
//     }
// }

void WOA::CheckBoundary()
{
    static Logger log(__FUNCTION__);
    for (int i = 0; i < this->population; ++i)
    {
        for (int j = 1; j < this->size-1; ++j)
        {
            bool isOutside = false;
            State* closestWaypoint;
            long double distance, shortestDistance = std::numeric_limits<long double>::max();
            State& current = this->particles[i].array[j];
            for (int k = 1; k < this->size-1; ++k)
            {
                State& original = this->rrtStarPath->array[k]; 
                if (current.x > (original.x + this->widthBound))
                    isOutside = true;
                else if (current.x < (original.x - this->widthBound))
                    isOutside = true;
                if (current.y > (original.y + this->heightBound))
                    isOutside = true;
                else if (current.y < (original.y - this->heightBound))
                    isOutside = true;
                distance = CalculateEuclideanDistance(current.x, current.y, original.x, original.y);
                if (distance < shortestDistance)
                {
                    shortestDistance = distance;
                    closestWaypoint = &original;
                }
            }
            if (isOutside)
            {
                current.x = closestWaypoint->x;
                current.y = closestWaypoint->y;
            }
        }
    }
}

void WOA::RenderParticles()
{
    static Logger log(__FUNCTION__);
    cv::Mat image(500, 500, CV_8UC1, cv::Scalar(255, 255, 255));
    image = myMap.colored.clone();
    /// Draw all particles.
    int x, y;
    for (int i = 0; i < this->population; ++i){
        for (int j = 1; j < this->size; ++j)
        {
            x = XConvertToPixel(this->particles[i].array[j-1].x);
            y = YConvertToPixel(this->particles[i].array[j-1].y);
            cv::Point p(x, y);
            x = XConvertToPixel(this->particles[i].array[j].x);
            y = YConvertToPixel(this->particles[i].array[j].y);
            cv::Point q(x, y);
            line(image, p, q, cv::Scalar(0x00, 0x00, 0xFF), 1, cv::LINE_AA);
        }
    }
    for (int j = 1; j < this->size; ++j)
    {
        x = XConvertToPixel(this->best->array[j-1].x);
        y = YConvertToPixel(this->best->array[j-1].y);
        cv::Point p(x, y);
        x = XConvertToPixel(this->best->array[j].x);
        y = YConvertToPixel(this->best->array[j].y);
        cv::Point q(x, y);
        line(image, p, q, cv::Scalar(0x00, 0xFF, 0x00), 2, cv::LINE_AA);
    }

    /// Goal configuration
    x = XConvertToPixel(this->best->array[this->size-1].x);
    y = YConvertToPixel(this->best->array[this->size-1].y);      
    cv::Point goal(x, y);
    circle(image, goal, 3, cv::Scalar(0, 0, 0), -1, cv::LINE_AA);

    /// Init configuration
    x = XConvertToPixel(this->best->array[0].x);
    y = YConvertToPixel(this->best->array[0].y);
    cv::Point init(x, y);
    circle(image, init, 3, cv::Scalar(0, 0, 0), -1, cv::LINE_AA);

    cv::imshow("RRT* WOA", image);
    cv::waitKey(0);
    myVideo->write(image);
}

void WOA::CleanUp()
{
    ;
}

/// ------------------------------------------------------------------------------------ 

long double WOA::GenerateRandom(long double a, long double b)
{
    std::random_device rd;
    std::default_random_engine engine(rd());
    std::uniform_real_distribution<long double> distribution(a, b);
    return distribution(engine);
}

int WOA::GenerateRandom(int a, int b)
{
    std::random_device rd;
    std::default_random_engine engine(rd());
    std::uniform_real_distribution<long double> distribution((long double)a, (long double)b);
    long double random = distribution(engine);
    return (int)(std::floor(random));  
}

long double WOA::CalculateEuclideanDistance(long double& x, long double& y, long double& a, long double& b){
    return sqrt(pow(x-a, 2)+pow(y-b, 2));
}

/// todo
long double WOA::CalculateDistance(long double& v){
    long double timeStep = mySettings["RRT*_time_step"];
    return (v * timeStep);
}
/// todo
long double WOA::CalculateAngle(long double& w){
    long double timeStep = mySettings["RRT*_time_step"];
    return (w * timeStep);
}

int WOA::XConvertToPixel(long double& x)
{
    return (int)((x + (-myMap.origin.x)) / myMap.resolution);
}

int WOA::YConvertToPixel(long double& y)
{
    return (int)((y + (-myMap.origin.x)) / myMap.resolution);
}

bool WOA::DoIntersect(cv::Point& p1, cv::Point& q1, cv::Point& p2, cv::Point& q2)
{
    int o1 = Orientation(p1, q1, p2);
    int o2 = Orientation(p1, q1, q2);
    int o3 = Orientation(p2, q2, p1);
    int o4 = Orientation(p2, q2, q1);
    if (o1 != o2 && o3 != o4)
        return true;
    if (o1 == 0 && OnSegment(p1, p2, q1)) return true;
    if (o2 == 0 && OnSegment(p1, q2, q1)) return true;
    if (o3 == 0 && OnSegment(p2, p1, q2)) return true;  
    if (o4 == 0 && OnSegment(p2, q1, q2)) return true;
    return false;
}

int WOA::Orientation(cv::Point& p, cv::Point& q, cv::Point& r)
{
    int val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
    if (val == 0) return 0;  // collinear
    return (val > 0)? 1: 2; // clock or counterclock wise
}

bool WOA::OnSegment(cv::Point& p, cv::Point& q, cv::Point& r)
{
    if (q.x <= std::max(p.x, r.x) && q.x >= std::min(p.x, r.x) && q.y <= std::max(p.y, r.y) && q.y >= std::min(p.y, r.y))
       return true;
    return false;
}

bool WOA::IsObstacleFree(State& sNew, State& sNeighbor)
{
    #ifdef DEBUG
        static Logger log("IsObstacleFree [Segment]x");
    #endif
    int pX = XConvertToPixel(sNew.x);
    int pY = YConvertToPixel(sNew.y);
    cv::Point p(pX, pY);
    int qX = XConvertToPixel(sNeighbor.x);
    int qY = YConvertToPixel(sNeighbor.y);
    cv::Point q(qX, qY);
    for (auto& segment: myMap.segments)
    {
        if (DoIntersect(p, q, segment.p, segment.q))
        {
            #ifdef DEBUG
                log.trace("Segment p(x:{:+.2f}, y:{:+.2f}) q(x:{:+.2f}, y:{:+.2f}) is not free", sNew.x, sNew.y, sNeighbor.x, sNeighbor.y);
            #endif
            return false;
        }
    }
    #ifdef DEBUG
        log.trace("Segment p(x:{:+.2f}, y:{:+.2f}) q(x:{:+.2f}, y:{:+.2f}) is free", sNew.x, sNew.y, sNeighbor.x, sNeighbor.y);
    #endif
    return true;
}

void WOA::SaveGreatest()
{
    for (int i = 0; i < this->size; ++i)
    {
        this->greatest.array[i] = this->best->array[i];
    }
}

void WOA::LoadGreatest()
{
    for (int i = 0; i < this->size; ++i)
    {
        this->best->array[i] = this->greatest.array[i];
    }
}