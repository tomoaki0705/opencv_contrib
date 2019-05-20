#ifndef __OPENCV_BDH_HPP__
#define __OPENCV_BDH_HPP__

#include <opencv2/core.hpp>

/**
* @defgroup bdh Bucket Distance Hash
* This module computes the nearest neighor distance
*/
namespace cv {
namespace bdh {
    typedef char featureElement;
    /////////////// Load Function ////////////////////////
    /**
    * @brief read point set
    * @return is file open ?
    * @param path path of binary file
    * @param dim  dimension of features
    * @param num  number of features
    * @param data actual output
    */
    CV_EXPORTS_W bool readBinary(const cv::String &path, unsigned &dim, unsigned &num, featureElement** data);

    /////////////// Search Function ////////////////////////
    //CV_EXPORTS_W 
    //    template <typename data_t>
    //int BDH<data_t>::NearestNeighbor(
    //    data_t* query,
    //    point_t<data_t>* point,
    //    double searchParam,
    //    search_mode searchMode,
    //    int K,
    //    double epsilon
    //);
}
}

#endif
