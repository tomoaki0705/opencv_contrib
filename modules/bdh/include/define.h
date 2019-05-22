/**
* @file		define.h
* @author	Tomokazu Sato
* @date		2015/05/05
*/

#ifndef __DEFINE__
#define __DEFINE__

#include <iostream>
using namespace std;

/**
* @brief norm for distance
* @return scalar distance
*/
inline double NORM(double x)
{
	return x*x;
}

///**
//* @brief type of layer. a node coresponde to a subspace
//*/
//struct layer_t
//{
//	int k;			//!< number of nodes
//	double restMin;	//!< the minimam rest distance from this layer
//	double restMax;	//!< the maximam rest distance from this layer
//	double gap;		//!< the gap of distance between max and min
//	node_t* node;	//!< nodes.
//
//	void calc_gap()
//	{
//		gap	= node[k - 1].distance - node[0].distance;
//	}
//
//	/**
//	* @brief compare the gap
//	* @return is which gap larger ?
//	*/
//	bool operator < (const layer_t& obj)
//	{
//		return gap > obj.gap;
//	}
//};
//
///**


///**
//* @brief status of neare bucket search
//*/
//struct hashKey_t{
//
//	size_t hashKey;	//!< hash value
//	double dist;	//!< distance
//
//	/**
//	* @brief default constructor
//	*/
//	hashKey_t()
//		:hashKey(0)
//		, dist(0)
//	{}
//
//	/**
//	* @brief constructor. allocate memory of nodeIdx and deep copied.
//	*/
//	hashKey_t(
//		size_t hashKey,	//!< hash value
//		double dist)	//!< distance
//		: hashKey(hashKey)
//		, dist(dist)
//	{}
//
//	void setVariable(size_t _hashKey, double _dist)
//	{
//		hashKey = _hashKey;
//		dist = _dist;
//	}
//};

#endif//__define__