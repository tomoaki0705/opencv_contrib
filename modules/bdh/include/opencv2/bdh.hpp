#ifndef __OPENCV_BDH_HPP__
#define __OPENCV_BDH_HPP__

#include <opencv2/core.hpp>

/**
* @defgroup bdh Bucket Distance Hash
* This module computes the nearest neighor distance
*/
namespace cv {
namespace bdh {
    ///////////// Search Function ////////////////////////
    CV_EXPORTS_W 
        template <typename data_t>
    int BDH<data_t>::NearestNeighbor(
        data_t* query,
        point_t<data_t>* point,
        double searchParam,
        search_mode searchMode,
        int K,
        double epsilon
    );
}
}

#endif
