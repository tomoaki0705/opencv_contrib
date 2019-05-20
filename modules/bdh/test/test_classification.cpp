/*
By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.

License Agreement
For Open Source Computer Vision Library
(3-clause BSD License)

Copyright (C) 2013, OpenCV Foundation, all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

* Neither the names of the copyright holders nor the names of the contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

bool loadFeature(String &filename, unsigned &dim, unsigned &num, featureElement** data)
{
    static string dataSetPath = TS::ptr()->get_data_path() + "bdh/";
    string filePath = dataSetPath + filename;
    return cv::bdh::readBinary(filePath, dim, num, data);
}

TEST(BDH_Classification, Load)
{
    unsigned int num, dim;
    char** data = NULL;
    bool readResult = loadFeature(String("sift10K.ucdat"), dim, num, data);
    EXPECT_TRUE(readResult);
    EXPECT_EQ(num, 10000);
    EXPECT_EQ(dim, 128);
    //delete data point
    for (size_t n = 0; n < num; n++)
    {
        delete[] data[n];
    }
    delete[] data;
    data = NULL;
    readResult = loadFeature(String("sift1K.ucquery"), dim, num, data);
    EXPECT_TRUE(readResult);
    EXPECT_EQ(num, 1000);
    EXPECT_EQ(dim, 128);
    //delete data point
    for (size_t n = 0; n < num; n++)
    {
        delete[] data[n];
    }
    delete[] data;
}

TEST(BDH_Classification, Classify)
{
    unsigned int num, dim;
    char** data = NULL;
    bool readResult = loadFeature(String("sift10K.ucdat"), dim, num, data);
    cv::bdh::Index bdh;
}

}} // namespace
