#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <inttypes.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdlib>
#include <iomanip>
#include <time.h>
#include <random>
#include <limits>
#include <cmath>
#include "rrt_star.hpp"
#include "SETTINGS.hpp"
#include "objects.hpp"
#include "logger.hpp"
#include "map.hpp"
#include "woa.hpp"

extern cv::VideoWriter* myVideo; /// Object used to write videos.
extern nlohmann::json mySettings;
extern Map myMap;


Path RRT_Star::Build(){
    #ifdef DEBUG
        static Logger log(__FUNCTION__);
        log.debug("Initialize()");
    #endif
    Initialize();
    #ifdef DEBUG
        log.debug("InsertRoot()");
    #endif
    InsertRoot();
    for (this->count = 1; this->count < this->iterations; ++(this->count))
    {
        #ifdef DEBUG
            log.debug("State {}: SampleFreeSpace()", this->count);
        #endif
        State sRand = SampleFreeSpace();
        #ifdef DEBUG
            log.debug("State {}: FindNearest()", this->count);
        #endif
        State* sNear = FindNearest(sRand);
        #ifdef DEBUG
            log.debug("State {}: Steer()", this->count);
        #endif
        State sNew = SteerUsingWOA(sNear, sRand);
        if (IsObstacleFree(sNew))
        {
            #ifdef DEBUG
                log.debug("State {}: GetNeighbors()", this->count);
            #endif
            std::vector<State*> neighbors = GetNeighbors(sNew);
            if (neighbors.empty())
            {
                #ifdef DEBUG
                    log.trace("No valid neighbors");
                #endif
                --(this->count);
                continue;
            }
            #ifdef DEBUG
                log.debug("State {}: ChooseParent()", this->count);
            #endif
            State* sParent = ChooseParent(sNew, neighbors);
            #ifdef DEBUG
                log.debug("State {}: Insert()", this->count);
            #endif
            State* psNew = Insert(sNew, sParent);
            #ifdef DEBUG
                log.debug("State {}: RewireTree()", this->count);
            #endif
            RewireTree(psNew, neighbors);
        }
        else{
            --(this->count);
            continue;
        }
    }
    #ifdef DEBUG
        log.debug("ShortestPath()");
    #endif
    return ShortestPath();
}

void RRT_Star::Initialize(){
    this->iterations = mySettings["RRT*_iterations"];
    this->timeStep = mySettings["RRT*_time_step"];
    this->linearVelocity = mySettings["RRT*_linear_velocity"];
    this->angularVelocity = mySettings["RRT*_angular_velocity"];
    this->searchRadius = mySettings["RRT*_search_radius"];
    this->maxSteeringAngle = mySettings["RRT*_max_steering_angle"];
    this->goalThreshold = mySettings["RRT*_goal_threshold"];
    this->states = new State[this->iterations];
    return;
}

void RRT_Star::InsertRoot(){
    states[0] = this->sInit;
    return;
}

State RRT_Star::SampleFreeSpace(){
    #ifdef DEBUG
        static Logger log(__FUNCTION__);
    #endif
    State sRand;
    sRand.x = GenerateRandom(myMap.origin.x);
    sRand.y = GenerateRandom(myMap.origin.y);
    #ifdef DEBUG
        log.trace("sRand (x:{:+.2f}, y:{:+.2f})", sRand.x, sRand.y);
    #endif
    return sRand;
}

State* RRT_Star::FindNearest(State& sRand)
{
    #ifdef DEBUG
        static Logger log(__FUNCTION__);
    #endif
    State *sNear;
    long double shortestDistance = std::numeric_limits<long double>::max();
    long double distance;
    for(int i = 0; i < this->count; ++i) {
        distance = CalculateEuclideanDistance(sRand.x, sRand.y, this->states[i].x, this->states[i].y);
        if (distance < shortestDistance){
            shortestDistance = distance;
            sNear = &states[i];
        }
    }
    #ifdef DEBUG
        log.trace("sNear (x:{:.2f}, y:{:.2f}, z:{:.2f})", sNear->x, sNear->y, sNear->z);
    #endif
    return sNear;
}

State RRT_Star::Steer(State* sNear, State& sRand)
{
    #ifdef DEBUG
        static Logger log(__FUNCTION__);
    #endif
    std::vector<long double> angularVelocities = angularVelocity.get<std::vector<long double>>();
    int N = angularVelocities.size();
    long double angVel[N], dist[N], x_temp[N], y_temp[N];
    // Copy angular velocities into the array.
    for (int i = 0; i < N; i++)
    {
        angVel[i] = angularVelocities[i];
    }
    // Calculate resulting states and their distance to 's_rand'.
    for (int i = 0; i < N; ++i)
    {
        x_temp[i] = (sNear->x) - CalculateDistance(this->linearVelocity) * sin(sNear->z + CalculateAngle(angVel[i]) / 2.0l);
        y_temp[i] = (sNear->y) + CalculateDistance(this->linearVelocity) * cos(sNear->z + CalculateAngle(angVel[i]) / 2.0l);
        dist[i] = CalculateEuclideanDistance(sRand.x, sRand.y, x_temp[i], y_temp[i]);
    }
    // Sort arrays by distance.
    for (int i = 0; i < N-1; ++i)
    {
        for (int j = 0; j < N-i-1; ++j)
        {
            if (dist[j] > dist[j+1])
            {
                Swap(&angVel[j], &angVel[j+1]);
                Swap(&dist[j], &dist[j+1]);
                Swap(&x_temp[j], &x_temp[j+1]);
                Swap(&y_temp[j], &y_temp[j+1]);
            }
        }
    }
    // Choose the closest as 'sNew'.
    State sNew;
    sNew.x = x_temp[0];
    sNew.y = y_temp[0];
    sNew.z =  sNear->z + CalculateAngle(angVel[0]);
    #ifdef DEBUG
        log.trace("sNew (x:{:+.2f}, y:{:+.2f}, z:{:+.2f})", sNew.x, sNew.y, sNew.z);
    #endif
    return sNew;
}

State RRT_Star::SteerUsingWOA(State* sNear, State& sRand)
{
    WOA woa(*sNear, sRand, this->sGoal);
    return woa.Apply();
}


bool RRT_Star::IsObstacleFree(State& sNew)
{
    #ifdef DEBUG
        static Logger log("IsObstacleFree [Point]");
    #endif
    int x = XConvertToPixel(sNew.x);
    int y = YConvertToPixel(sNew.y);
    cv::Point p(x, y);
    for (auto& segment: myMap.segments)
    {
        if (DoIntersect(p, p, segment.p, segment.q))
        {
            #ifdef DEBUG
                log.trace("State (x:{:+.2f}, y:{:+.2f}) is not free", sNew.x, sNew.y);
            #endif
            return false;
        }
    }
    #ifdef DEBUG
        log.trace("State (x:{:+.2f}, y:{:+.2f}) is free", sNew.x, sNew.y);
    #endif
    return true;
}

bool RRT_Star::IsObstacleFree(State& sNew, State& sNeighbor)
{
    #ifdef DEBUG
        static Logger log("IsObstacleFree [Segment]");
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

std::vector<State*> RRT_Star::GetNeighbors(State& sNew)
{
    #ifdef DEBUG
        static Logger log(__FUNCTION__);
    #endif
    std::vector<State*> neighbors;
    for (int i = 0; i < this->count; ++i)
    {
        long double distance = CalculateEuclideanDistance(sNew.x, sNew.y, this->states[i].x, this->states[i].y);
        if (distance < this->searchRadius){
            if (fabs(sNew.z - this->states[i].z) < this->maxSteeringAngle)
            {
                if (IsObstacleFree(sNew, this->states[i]))
                {
                    neighbors.push_back(&this->states[i]);
                }
            }
        }
    }
    return neighbors;
}

State* RRT_Star::ChooseParent(State& sNew, std::vector<State*>& neighbors)
{
    State* sParent;
    long double lowestCost = std::numeric_limits<long double>::max();
    for (auto& neighbor: neighbors)
    {
        long double cost = neighbor->c + CalculateCost(sNew, *neighbor);
        if (cost < lowestCost)
        {
            lowestCost = cost;
            sParent = neighbor;
        }
    }
    return sParent;
}

State* RRT_Star::Insert(State& sNew, State* sParent)
{
    Velocity velocity = InverseOdometry(sNew, *sParent);
    long double cost = CalculateCost(sNew, *sParent);
    sNew.p = sParent;
    sNew.v = velocity.v;
    sNew.w = velocity.w;
    sNew.c = sParent->c + cost;
    this->states[this->count] = sNew;
    return &this->states[this->count];
}

void RRT_Star::RewireTree(State* sNew, std::vector<State*>& neighbors)
{
    #ifdef DEBUG
        static Logger log(__FUNCTION__);
    #endif
    State* sNewParent = (*sNew).p;
    neighbors.erase(std::remove(neighbors.begin(), neighbors.end(), sNewParent), neighbors.end());
    for (auto& neighbor: neighbors)
    {
        double cost = sNew->c + CalculateCost(*sNew, *neighbor);
        if (cost < neighbor->c)
        {
            State* currentParent = neighbor->p;
            State* temp = neighbor->p = sNew;
            for (int i = 0; i < 3; ++i)
            {
                if (temp)
                {
                    temp = temp->p;
                }
            }
            if (temp == sNew) // If cycle happened. Revert to old parent.
            {
                neighbor->p = currentParent;
                continue;
            }
            neighbor->c = cost;
            Velocity velocity = InverseOdometry(*neighbor, *sNew);
            neighbor->v = velocity.v;
            neighbor->w = velocity.w;
        }
    }
    return;
}

void RRT_Star::Render(Path& path){
    myMap.colored = cv::Mat(500, 500, CV_8UC3, cv::Scalar(0,0,0));
    int x, y;
    /// Draw all branches.
    for (int i = 1; i < this->count; ++i)
    {
        State a = this->states[i];
        State b = *(a.p);
        x = XConvertToPixel(a.x);
        y = YConvertToPixel(a.y);
        cv::Point p(x, y); 
        x = XConvertToPixel(b.x);
        y = YConvertToPixel(b.y);
        cv::Point q(x, y);
        line(myMap.colored, p, q, cv::Scalar(0xBC, 0x44, 0x39), 1, cv::LINE_AA);
    }

    /// Initial configuration.
    x = XConvertToPixel(this->sInit.x);
    y = YConvertToPixel(this->sInit.y);
    cv::Point init(x, y);
    circle(myMap.colored, init, 9, cv::Scalar(0x00, 0x00, 0x00), -1, cv::LINE_AA);
    circle(myMap.colored, init, 7, cv::Scalar(0x4A, 0xC0, 0x03), -1, cv::LINE_AA);

    /// Goal threshold region and configuration.
    x = XConvertToPixel(this->sGoal.x);
    y = YConvertToPixel(this->sGoal.y);
    cv::Point goal(x, y);
    int goal_radius = (int)(this->goalThreshold / myMap.resolution);
    circle(myMap.colored, goal, goal_radius+2, cv::Scalar(0x00, 0x00, 0x00), -1, cv::LINE_AA);
    circle(myMap.colored, goal, goal_radius, cv::Scalar(0x39, 0x29, 0xED), -1, cv::LINE_AA);

    /// Draw shortest path.
    for (int i = 1; i < path.size; ++i)
    {
        State a = path.array[i];
        State b = *(a.p);
        x = XConvertToPixel(a.x);
        y = YConvertToPixel(a.y);
        cv::Point p(x, y); 
        x = XConvertToPixel(b.x);
        y = YConvertToPixel(b.y);
        cv::Point q(x, y); 
        line(myMap.colored, p, q, cv::Scalar(0x00, 0x00, 0x00), 5, cv::LINE_AA);
        line(myMap.colored, p, q, cv::Scalar(0x4A, 0xC0, 0x03), 3, cv::LINE_AA);
    }
    
    /// Draw obstacle segments.
    for (auto& segment: myMap.segments)
    {
        line(myMap.image, segment.p, segment.q, cv::Scalar(0x97), 1, cv::LINE_AA);
    }
    cv::imshow("RRT* WOA", myMap.colored);
}

Path RRT_Star::ShortestPath(){
    #ifdef DEBUG
        static Logger log(__FUNCTION__);
    #endif
    double lowestCost = std::numeric_limits<long double>::max();
    State* bestState = nullptr;
    /// Find the state that lies within goal region and has the lowest cost.
    for (int i = 0; i < this->count; ++i)
    {
        long double distanceFromGoal = CalculateEuclideanDistance(this->states[i].x, this->states[i].y, this->sGoal.x, this->sGoal.y);
        if (distanceFromGoal < this->goalThreshold)
        {
            if (this->states[i].c < lowestCost)
            {
                lowestCost = this->states[i].c;
                bestState = &this->states[i];
            }
        }
    }
    /// If no path found, return.
    Path path;
    if (!bestState)
    {
        #ifdef DEBUG
            log.error("No path found");
        #endif
        exit(1);
    }
    #ifdef DEBUG
        log.trace("Best State: (x:{:+.2f}, y:{:+.2f}, z:{:+.2f}, c:{:+.2f})",bestState->x, bestState->y, bestState->z, bestState->c);
    #endif
    /// Find how many state constitute the shortest path.
    int i = 1;
    State* pointer = bestState;
    while (pointer->p){
        ++i;
        pointer = pointer->p;
    }

    /// Copy the states into the path array.
    pointer = bestState;
    path.size = i;
    path.array = new State[i];
    for (int j = i-1; j >= 0; --j){
        path.array[j] = *pointer;
        pointer = pointer->p;
    }
    #ifdef DEBUG
        /// Log the complete path.
        for (int i = 0; i < path.size; ++i){
            log.trace("State {}: (x:{:+.2f}, y:{:+.2f}, z:{:+.2f}, v:{:+.2f}, w:{:+.2f}, c:{:+.2f})", i, path.array[i].x, path.array[i].y, path.array[i].z, path.array[i].v, path.array[i].w, path.array[i].c);
        }
    #endif
    return path;
}

void RRT_Star::CleanUp(){
    delete this->states;
    return;
}











//**********************************************************************
// This is utility section
//**********************************************************************        











Velocity RRT_Star::InverseOdometry(State& sNew, State& sOld){
    long double deltaX = sNew.x - sOld.x;
    long double deltaY = sNew.y - sOld.y;
    long double deltaZ = sNew.z - sOld.z;
    long double distance = sqrt(pow(deltaX, 2)+pow(deltaY, 2));
    long double linVel = distance / this->timeStep;
    long double angVel = deltaZ / this->timeStep;
    Velocity velocity;
    velocity.v = linVel;
    velocity.w = angVel;
    return velocity;
}

long double RRT_Star::CalculateEuclideanDistance(long double& x, long double& y, long double& a, long double& b){
    return sqrt(pow(x-a, 2)+pow(y-b, 2));
}

long double RRT_Star::CalculateDistance(long double& v){
    return (v * this->timeStep);
}

long double RRT_Star::CalculateAngle(long double& w){
    return (w * this->timeStep);
}

void RRT_Star::Swap(long double *a, long double *b){
    double temp = *a;
    *a = *b;
    *b = temp;
    return;
}

long double RRT_Star::CalculateCost(State& a, State& b){
    return CalculateEuclideanDistance(a.x, a.y, b.x, b.y);
}

int RRT_Star::Orientation(cv::Point& p, cv::Point& q, cv::Point& r){
    int val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
    if (val == 0) return 0;  // collinear
    return (val > 0)? 1: 2; // clock or counterclock wise
}

bool RRT_Star::OnSegment(cv::Point& p, cv::Point& q, cv::Point& r){
    if (q.x <= std::max(p.x, r.x) && q.x >= std::min(p.x, r.x) && q.y <= std::max(p.y, r.y) && q.y >= std::min(p.y, r.y))
       return true;
    return false;
}

int RRT_Star::XConvertToPixel(long double& x){
    return (int)((x + (-myMap.origin.x)) / myMap.resolution);
}

int RRT_Star::YConvertToPixel(long double& y){
    return (int)((y + (-myMap.origin.y)) / myMap.resolution);
}

bool RRT_Star::DoIntersect(cv::Point& p1, cv::Point& q1, cv::Point& p2, cv::Point& q2){
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

long double RRT_Star::GenerateRandom(long double& a){
    long double lowerBound = a;
    long double upperBound = -a;
    std::default_random_engine re{std::random_device{}()};
    std::uniform_real_distribution<long double> unif(lowerBound, upperBound);
    return unif(re);
}