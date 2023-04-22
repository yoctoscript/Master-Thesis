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
#include "map.hpp"

extern nlohmann::json mySettings;
extern Map myMap;

Path RRT_Star::Build(){
    static Logger log(__FUNCTION__);
    log.debug("Initialize()");
    Initialize();
    log.debug("InsertRoot()");
    InsertRoot();
    for (this->count = 1; this->count < this->iterations; ++(this->count)){
        log.debug("State {}: SampleFreeSpace()", this->count);
        State sRand = SampleFreeSpace();
        log.debug("State {}: FindNearest()", this->count);
        State* sNear = FindNearest(sRand);
        log.debug("State {}: Steer()", this->count);
        State sNew = Steer(sNear, sRand);
        if (IsObstacleFree(sNew)){
            log.debug("State {}: GetNeighbors()", this->count);
            std::vector<State*> neighbors = GetNeighbors(sNew);
            if (neighbors.empty()){
                log.trace("No valid neighbors");
                --(this->count);
                continue;
            }
            log.debug("State {}: ChooseParent()", this->count);
            State* sParent = ChooseParent(sNew, neighbors);
            log.debug("State {}: Insert()", this->count);
            State* psNew = Insert(sNew, sParent);
            log.debug("State {}: RewireTree()", this->count);
            RewireTree(psNew, neighbors);
        }
        else{
            --(this->count);
            continue;
        }
    }
    log.debug("ShortestPath()");
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
    static Logger log(__FUNCTION__);
    State sRand;
    sRand.x = GenerateRandom(myMap.origin.x);
    sRand.y = GenerateRandom(myMap.origin.y);
    log.trace("sRand (x:{:+.2f}, y:{:+.2f})", sRand.x, sRand.y);
    return sRand;
}

State* RRT_Star::FindNearest(State& sRand){
    static Logger log(__FUNCTION__);
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
    log.trace("sNear (x:{:.2f}, y:{:.2f}, z:{:.2f})", sNear->x, sNear->y, sNear->z);
    return sNear;
}

State RRT_Star::Steer(State* sNear, State& sRand){
    static Logger log(__FUNCTION__);
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
    log.trace("sNew (x:{:+.2f}, y:{:+.2f}, z:{:+.2f})", sNew.x, sNew.y, sNew.z);
    return sNew;
}

bool RRT_Star::IsObstacleFree(State& sNew)
{
    static Logger log("IsObstacleFree [Point]");
    int x = XConvertToPixel(sNew.x);
    int y = YConvertToPixel(sNew.y);
    cv::Point p(x, y);
    for (auto& segment: myMap.segments)
    {
        if (DoIntersect(p, p, segment.p, segment.q))
        {
            log.trace("State (x:{:+.2f}, y:{:+.2f}) is not free", sNew.x, sNew.y);
            return false;
        }
    }
    log.trace("State (x:{:+.2f}, y:{:+.2f}) is free", sNew.x, sNew.y);
    return true;
}

bool RRT_Star::IsObstacleFree(State& sNew, State& sNeighbor)
{
    static Logger log("IsObstacleFree [Segment]");
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
            log.trace("Segment p(x:{:+.2f}, y:{:+.2f}) q(x:{:+.2f}, y:{:+.2f}) is not free", sNew.x, sNew.y, sNeighbor.x, sNeighbor.y);
            return false;
        }
    }
    log.trace("Segment p(x:{:+.2f}, y:{:+.2f}) q(x:{:+.2f}, y:{:+.2f}) is free", sNew.x, sNew.y, sNeighbor.x, sNeighbor.y);
    return true;
}

std::vector<State*> RRT_Star::GetNeighbors(State& sNew)
{
    static Logger log(__FUNCTION__);
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
    static Logger log(__FUNCTION__);
    State* sNewParent = (*sNew).p;
    neighbors.erase(std::remove(neighbors.begin(), neighbors.end(), sNewParent), neighbors.end());
    for (auto& neighbor: neighbors)
    {
        double cost = sNew->c + CalculateCost(*sNew, *neighbor);
        if (cost < neighbor->c)
        {
            Velocity velocity = InverseOdometry(*neighbor, *sNew);
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
            neighbor->v = velocity.v;
            neighbor->w = velocity.w;
        }
    }
    return;
}

void RRT_Star::Render(Path& path){
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
        line(myMap.image, p, q, cv::Scalar(0x97), 1, cv::LINE_AA);
    }

    /// Initial configuration.
    x = XConvertToPixel(this->sInit.x);
    y = YConvertToPixel(this->sInit.y);
    cv::Point init(x, y);
    circle(myMap.image, init, 5, cv::Scalar(0x3B), -1, cv::LINE_AA);

    /// Goal threshold region and configuration.
    x = XConvertToPixel(this->sGoal.x);
    y = YConvertToPixel(this->sGoal.y);
    cv::Point goal(x, y);
    int goal_radius = (int)(this->goalThreshold / myMap.resolution);
    circle(myMap.image, goal, goal_radius, cv::Scalar(0x97), -1, cv::LINE_AA);
    circle(myMap.image, goal, 5, cv::Scalar(0x3B), -1, cv::LINE_AA);

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
        line(myMap.image, p, q, cv::Scalar(0x97), 4, cv::LINE_AA);
    }
    
    /// Draw obstacle segments.
    for (auto& segment: myMap.segments)
    {
        line(myMap.image, segment.p, segment.q, cv::Scalar(0x97), 1, cv::LINE_AA);
    }
    myDebug ? cv::imshow("RRT*", myMap.image) : (void)0;
}

Path RRT_Star::ShortestPath(){
    static Logger log(__FUNCTION__);
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
        log.error("No path found");
        exit(1);
    }
    log.trace("Best State: (x:{:+.2f}, y:{:+.2f}, z:{:+.2f}, c:{:+.2f})",bestState->x, bestState->y, bestState->z, bestState->c);
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
    /// Log the complete path.
    for (int i = 0; i < path.size; ++i){
        log.trace("State {}: (x:{:+.2f}, y:{:+.2f}, z:{:+.2f}, v:{:+.2f}, w:{:+.2f}, c:{:+.2f})", i, path.array[i].x, path.array[i].y, path.array[i].z, path.array[i].v, path.array[i].w, path.array[i].c);
    }
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