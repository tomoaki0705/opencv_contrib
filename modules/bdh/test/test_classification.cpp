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

    const cv::String kFeatureFilename = "sift10K.ucdat";
    const cv::String kQueryFilename = "sift1K.ucquery";
    const cv::String kPcaFilename = "pca.dat";
    const cv::String kParameterFilename = "parameter.bdh";
    const cv::String kHashTableFilename = "BDHtable.tbl";

#define MAKE_FULL_PATH(path) TS::ptr()->get_data_path() + "bdh/" + path

bool loadFeature(const String &filename, Mat& data)
{
    String filePath = MAKE_FULL_PATH(filename);
    unsigned int dim, num;
    bool result = cv::bdh::readBinary(filePath, dim, num, data);
    data = cv::Mat(num, dim, CV_8UC1);
    if (result == true)
    {
        for (size_t y = 0; y < num; y++)
        {
            memcpy((void*)(data.data + y * data.step), data.row(y).data, sizeof(featureElement)*dim);
        }
    }
    return result;
}

bool loadFeature(const cv::String &filename, unsigned &dim, unsigned &num, Mat &data)
{
    static String dataSetPath = TS::ptr()->get_data_path() + "bdh/";
    String filePath = dataSetPath + filename;
    return cv::bdh::readBinary(filePath, dim, num, data);
}

TEST(BDH_Classification, Load)
{
    unsigned int num, dim;
    Mat data;
    bool readResult = loadFeature(kFeatureFilename, dim, num, data);
    EXPECT_TRUE(readResult);
    EXPECT_EQ(num, (unsigned int)10000);
    EXPECT_EQ(dim, (unsigned int)128);

    readResult = loadFeature(kQueryFilename, dim, num, data);
    EXPECT_TRUE(readResult);
    EXPECT_EQ(num, (unsigned int)1000);
    EXPECT_EQ(dim, (unsigned int)128);

}

TEST(BDH_Classification, Classify)
{
    unsigned int num, dim;
    featureElement **data = NULL;
    cv::Mat query, matData;
    bool readResult = loadFeature(kFeatureFilename, matData);
    EXPECT_TRUE(readResult);
    cv::bdh::Index bdh;
#if 1
    bdh.loadParameters(MAKE_FULL_PATH(kParameterFilename));
    bdh.loadTable(MAKE_FULL_PATH(kHashTableFilename));
#else
    bdh.Build(matData);
#endif

    double searchParam = static_cast<unsigned>(bdh.get_nDdataPoints()*0.001);
    cout << "read query point set." << endl;
    unsigned nQuery;
    loadFeature(kQueryFilename, dim, nQuery, query);

    int* NNC = new int[nQuery];
    point_t** KNNpoint = new point_t*[nQuery];
    for (unsigned q = 0; q < nQuery; ++q)
    {
        KNNpoint[q] = new point_t[1];
    }

    double startTime = getTickCount();
    for (unsigned q = 0; q < nQuery; ++q)
    {
        NNC[q] = bdh.NearestNeighbor(query.row(q), KNNpoint[q], searchParam, bdh::search_mode::NumPoints, 1, DBL_MAX);
    }
    double endTime = getTickCount();

    std::vector<int> reference;
    bool classResult = readCorrectClass(TS::ptr()->get_data_path() + "bdh/correctClass.txt", reference);


    for (unsigned qv = 0; qv < nQuery; ++qv)
    {
        EXPECT_EQ(NNC[qv], reference[qv]);
    }
    cout << "average query time : " << ((endTime - startTime) / (cv::getTickFrequency() / 1000.)) / nQuery << " ms" << endl;

    
}

}} // namespace
