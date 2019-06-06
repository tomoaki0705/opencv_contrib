#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/bdh.hpp"
#include "opencvPCA.h"
#include "Subspace.h"
#include <BDHtraining.h>
#include <BDH.h>
#include <limits>
const double deltaRate = 50;

///************** Indexing *******************/
namespace cv {
namespace bdh {
int Subspace::dim;

void Subspace::setNodeParam(node_t * node, InputArray _query) const
{
    Mat query = _query.getMat().reshape(1, 1);

    double* PCAquery = new double[subDim];
    getPCAdata(query, PCAquery);

    for (int b = 0; b < subHashSize; ++b)
    {
        node[b].distance = getDistanceToCentroid(PCAquery, b);
        node[b].hashKey = hashKey[b];
    }
    std::sort(node, node + subHashSize);

    delete[] PCAquery;
}

void Index::setLayerParam(layer_t * layer, InputArray _query) const
{

    layer_t* layer_p = layer;
    layer_t* layer_p_end = layer + M;
    for (int m = 0; layer_p != layer_p_end; ++m, ++layer_p)
    {
        layer_p->k = subspace[m].subHashSize;
        subspace[m].setNodeParam(layer_p->node, _query);
        layer_p->calc_gap();
    }
    std::sort(layer, layer + M);

    //m番目の部分空間からの残り距離の最大と最小を計算
    double distRestMin = 0;
    double distRestMax = 0;
    layer_p_end = layer - 1;
    for (layer_p = layer + M - 1; layer_p != layer_p_end; --layer_p)
    {
        layer_p->restMin = distRestMin += layer_p->node[0].distance;
        layer_p->restMax = distRestMax += layer_p->node[layer_p->k - 1].distance;
    }

}


int Index::NearBucket_R(
    const double Radius,//探索半径
    layer_t* const layer,//クエリから求めたレイヤごとの部分距離情報
    const status_t& status,//ノードの状態を表す
    vector<hashKey_t>& bucketList //![out] collect hash key of buckets near than Radius from query
)const
{
    const int m_plus1 = status.m + 1;

    int count = 0;

    if (m_plus1 == M)
    {
        size_t hashKey;
        index_t collision;
        node_t* node = layer[status.m].node;
        const double layerBound = Radius - status.dist;
        for (; node->distance <= layerBound; ++node)
        {
            hashKey = status.hashKey + node->hashKey;

            collision = hashTable.getCollision(hashKey);

            if (collision > 0)
            {
                count += collision;
                bucketList.push_back(
                    hashKey_t(hashKey, status.dist + node->distance)
                );
            }
        }
    }
    else
    {
        status_t statusNext(m_plus1);
        node_t* node = layer[status.m].node;

        const double layerBound = Radius - status.dist - layer[m_plus1].restMin;
        for (; node->distance <= layerBound; ++node)
        {
            statusNext.hashKey = status.hashKey + node->hashKey;
            statusNext.dist = status.dist + node->distance;

            count += NearBucket_R(
                Radius,
                layer,
                statusNext,
                bucketList
            );
        }
    }

    return count;
}


int Index::NearBucket_C(
    const double& Lbound,//探索下限
    const double& Ubound,//探索上限
    layer_t* const layer,//クエリから求めたレイヤごとの部分距離情報
    const status_t& status,
    vector<hashKey_t>& bucketList
) const
{

    const int m_plus1 = status.m + 1;
    int count = 0;

    if (m_plus1 == M)
    {
        size_t hashKey;
        hashKey_t Key;
        address_t bucket_p;

        const double layerLowerBound = Lbound - status.dist;
        const double layerUpperBound = Ubound - status.dist;
        node_t* node = layer[status.m].node;
        for (; node->distance <= layerLowerBound; ++node) {}
        for (; node->distance <= layerUpperBound; ++node)
        {
            hashKey = status.hashKey + node->hashKey;
            bucket_p = hashTable.getPointer(hashKey);
            if (bucket_p)
            {
                Key.setVariable(hashKey, status.dist + node->distance);
                bucketList.push_back(Key);
                count += *reinterpret_cast<collision_t*>(bucket_p);
            }
        }
    }
    else
    {
        const double layerLowerBound = Lbound - status.dist - layer[m_plus1].restMax;
        const double layerUpperBound = Ubound - status.dist - layer[m_plus1].restMin;

        status_t statusNext(m_plus1);

        node_t* node = layer[status.m].node;
        for (; node->distance <= layerLowerBound; ++node) {}
        for (; node->distance <= layerUpperBound; ++node)
        {
            statusNext.hashKey = status.hashKey + node->hashKey;
            statusNext.dist = status.dist + node->distance;

            count += NearBucket_C(
                Lbound, Ubound,
                layer, statusNext, bucketList
            );

        }

    }

    return count;
}


int Index::NearBucket_C_list(
    const double Rbound,//探索半径
    layer_t* const layer,//クエリから求めたレイヤごとの部分距離情報
    std::list<status_t>& statusQue,//探索途中のノードを保持
    std::list<status_t>::iterator* itr, //ノードの状態を表す
    std::vector<hashKey_t>& bucketList
) const
{

    const int m = (*itr)->m;
    const int m_plus1 = m + 1;
    node_t* const node = layer[m].node;

    int count = 0;
    double layerBound = Rbound - (*itr)->dist;
    int i = (*itr)->nodeIdx;
    if (m_plus1 == M)
    {
        size_t hashKey;
        index_t collision;

        for (; node[i].distance <= layerBound; ++i)
        {
            hashKey = (*itr)->hashKey + node[i].hashKey;
            collision = hashTable.getCollision(hashKey);
            if (collision > 0)
            {
                bucketList.push_back(
                    hashKey_t(hashKey, (*itr)->dist + node[i].distance)
                );
                count += collision;
            }
        }
    }
    else
    {
        layerBound -= layer[m_plus1].restMin;

        status_t statusNext(m_plus1);
        list<status_t>::iterator itr_next;
        for (; node[i].distance <= layerBound; ++i) {

            statusNext.hashKey = (*itr)->hashKey + node[i].hashKey;
            statusNext.dist = (*itr)->dist + node[i].distance;
            statusQue.push_front(statusNext);

            itr_next = statusQue.begin();
            count += NearBucket_C_list(
                Rbound, layer, statusQue, &itr_next, bucketList
            );
        }
    }

    //すべてのノードにアクセスしたか
    if (i == layer[m].k)
    {
        //ノードを消して次へ
        statusQue.erase((*itr)++);
    }
    else
    {
        //ノードの状態を更新して次へ
        (*itr)->nodeIdx = i;
        ++(*itr);
    }

    return count;
}

template <typename data_t>
double computeNorm(Mat& query, const data_t *data, double cutoffDistance)
{
    double distance = 0.0;
    for (int i = 0; i < query.rows && distance < cutoffDistance; i++)
    {
        distance += NORM(query.data[i] - (*data++));
    }
    return distance;
}

#define COMPUTE_NORM(type2, src1, src2, cutoff) computeNorm<type2>(src1, (type2*)src2, cutoff)

int Index::searchInBucket(
    InputArray _query,
    size_t hashKey,
    std::priority_queue<point_t>& NNpointQue
) const {
    CV_Assert((_query.getMat().rows == 1 && _query.getMat().cols == dim) || (_query.getMat().rows == dim && _query.getMat().cols == 1));

    bin_t bin;
    hashTable.getBin(hashKey, bin);

    /*set point's pointer*/
    collision_t coll = bin.collision;
    address_t addr = bin.addressOfChainList;
    address_t addr_end = addr + coll*entrySize;
    double KNNdist = NNpointQue.top().distance;
    Mat query = _query.getMat().reshape(1, dim);

    while (addr != addr_end)
    {
        /*Distance Calculation*/
        double dist = 0.0;
        switch (query.depth())
        {
        case CV_8U:  dist = COMPUTE_NORM(unsigned char,  query, addr, KNNdist); break;
        case CV_8S:  dist = COMPUTE_NORM(signed char,    query, addr, KNNdist); break;
        case CV_16U: dist = COMPUTE_NORM(unsigned short, query, addr, KNNdist); break;
        case CV_16S: dist = COMPUTE_NORM(short,          query, addr, KNNdist); break;
        case CV_32S: dist = COMPUTE_NORM(int,            query, addr, KNNdist); break;
        case CV_32F: dist = COMPUTE_NORM(float,          query, addr, KNNdist); break;
        case CV_64F: dist = COMPUTE_NORM(double,         query, addr, KNNdist); break;
        default:
            CV_Error(Error::StsUnsupportedFormat, "unsupported type");
            break;
        }

        if (dist < KNNdist)
        {
            NNpointQue.pop();

            NNpointQue.push(
                point_t(
                    *reinterpret_cast<index_t*>(addr + pointSize),
                    dist)
            );

            KNNdist = NNpointQue.top().distance;
        }
        addr += entrySize;
    }

    return coll;
}

///////////// Search Function ////////////////////////
void Index::linearSearchInNNcandidates(InputArray _query, point_t * point, int K, double epsilon, std::vector<hashKey_t>& bucketList) const
{
    //生成されたハッシュキーを元にバケットを参照して最近傍点を探索 start//
    priority_queue<point_t> NNpointQue;
    //最近傍点保持用のヒープ木を初期化
    for (int i = 0; i < K; ++i)
    {
        NNpointQue.push(point_t(ULLONG_MAX, epsilon));
    }

    //見つけてきたハッシュキーを参照して最近傍点を探索する
    vector<hashKey_t>::iterator keyList_itr = bucketList.begin();
    vector<hashKey_t>::iterator keyList_itr_end = bucketList.end();
    for (; keyList_itr != keyList_itr_end; ++keyList_itr)
    {
        searchInBucket(_query, (*keyList_itr).hashKey, NNpointQue);
    }
    //生成されたハッシュキーを元にバケットを参照して最近傍点を探索 end//

    //優先度付きキュー内の最近傍点を返却用引数にコピー
    for (int i = K - 1; i >= 0; --i)
    {
        point[i] = NNpointQue.top();
        NNpointQue.pop();
    }
}



int Index::NearestNeighbor(
    InputArray _query,
    point_t* point,
    double searchParam,
    search_mode searchMode,
    int K,
    double epsilon
)const
{

    // Generate the Bucket Hash key of the needle (begin)//
    vector<hashKey_t> bucketList;
    int NNC = getBucketList(_query, searchParam, searchMode, bucketList);

    // Generate the Bucket Hash key of the needle (end)//
    linearSearchInNNcandidates(_query, point, K, epsilon, bucketList);

    return NNC;// Number of points used to compute the distance
}

bool readCorrectClass(const String& filename, std::vector<int>& correctClass)
{
    ifstream ifs(filename, ios::in);
    if (ifs.is_open() == false)
    {
        return false;
    }
    correctClass.clear();

    while (ifs.eof() == false)
    {
        int index, classNumber;
        ifs >> index;
        ifs >> classNumber;
        assert(correctClass.size() == index);
        correctClass.push_back(classNumber);
    }

    ifs.close();
    return true;
}

bool readBinary(const String &path, unsigned &dim, unsigned &num, OutputArray data, int type)
{
    ifstream ifs(path, ios::in | ios::binary);
    if (ifs.is_open() == false)
    {
        return false;
    }

    {
        uint32_t dimension, dataSize;
        ifs.read((char*)&dimension, sizeof(dimension));
        ifs.read((char*)&dataSize, sizeof(dataSize));
        dim = dimension;
        num = dataSize;
    }

    data.create(num, dim, type);
    Mat stub = data.getMat();

    //const int dSize = sizeof(featureElement)*dim;
    for (size_t n = 0; n < num; n++)
    {
        ifs.read((char*)stub.row((int)n).data, CV_ELEM_SIZE(type)*dim);
    }
    ifs.close();

    return true;
}

template <typename data_t>
void parameterTuning_ICCV2013(int dim, index_t num, data_t ** const data, base_t * const base, int P, int bit, int &M, size_t &hashSize, size_t &pointSize, size_t &entrySize, double &variance, HashTable &hashTable, double &delta, std::vector<Subspace> &subspace, Subspace& lestspace, double bit_step = 1.0, double sampling_rate = 1.0)
{
    hashSize = (size_t(1) << bit);//hash size is 2^bit
    Subspace::dim = dim;

    pointSize = sizeof(featureElement)*dim;//byte size of a point's value
    entrySize = pointSize + sizeof(index_t);//byte size to entry a point into hash table

    variance = 0;
    for (int d = 0; d < dim; ++d)
    {
        variance += base[d].variance;   // pca.eigenvalues
    }
    delta = variance / deltaRate;

    hashTable.initialize(entrySize, hashSize);

    BDHtraining<featureElement> BDHtrainer;

    if (sampling_rate < 1.0)
    {//use a part of data set for training

        data_t** l_data = new data_t*[num];
        index_t l_num = 0;

        double tmp = 0.0;
        for (size_t n = 0; n < num; ++n)
        {
            tmp += sampling_rate;
            if (sampling_rate >= 1.0)
            {
                l_data[l_num++] = data[n];
                sampling_rate -= 1.0;
            }
        }

        BDHtrainer.training_ICCV2013(
            dim, l_num, (featureElement**)l_data,
            base, P, bit, bit_step
        );

        delete[] l_data;

    }
    else
    {

        BDHtrainer.training_ICCV2013(
            dim, num, (featureElement**)data,
            base, P, bit, bit_step
        );

    }

    M = BDHtrainer.getM();
    const baseset_t* const baseSet = BDHtrainer.getBaseSet();
    const baseset_t& lestSet = BDHtrainer.getLestSet();

    size_t rank = 1;

    for (int m = 0; m < M; ++m)
    {
        Subspace stub;
        stub.setParameters(baseSet[m]);

        for (int i = 0; i < stub.subHashSize; ++i)
        {
            stub.hashKey.push_back(rank*i);
        }

        rank *= stub.subHashSize;
        subspace.push_back(stub);
    }

    lestspace.subDim = lestSet.subDim;
    lestspace.variance = lestSet.variance;

    std::vector<double> stubCentroid;
    for (int d = 0; d < lestspace.subDim; ++d)
    {
        stubCentroid.push_back(lestSet.base[d].mean);
        Mat stub(1, dim, CV_64FC1, lestSet.base[d].direction);
        lestspace.baseVector.push_back(stub);
    }
    lestspace.centroidVector.push_back(stubCentroid);

}

void Index::Build(InputArray data, PCA::Flags order)
{
    cv::Mat _data = data.getMat();
    originalData = _data.clone();
    cv::PCA pca(originalData, Mat(), order, order == PCA::DATA_AS_ROW ? originalData.rows : originalData.cols);

    int length = originalData.rows;
    dim = originalData.cols;

    // copy PCA direction to base_t for BDH
    base_t* base = new base_t[dim];
    for (int d = 0; d < dim; ++d)
    {
        base[d].mean = (double)pca.mean.at<float>(d);
        base[d].variance = pca.eigenvalues.at<float>(d);
        base[d].direction = new double[dim];    // pca.eigenvectors
        for (int x = 0; x < dim; x++)
        {
            base[d].direction[x] = (double)((float*)(pca.eigenvectors.data + d * pca.eigenvectors.step))[x];
        }
    }
    featureElement **convertData = new featureElement*[length];
    for (int n = 0; n < length; n++)
    {
        convertData[n] = new featureElement[dim];
        memcpy(convertData[n], (featureElement*)originalData.data + n * originalData.step, sizeof(featureElement) * dim);
    }

    parameterTuning_ICCV2013(dim, length, convertData, base, P, 13, M, hashSize, pointSize, entrySize, variance, hashTable, delta, subspace, lestspace, 0.1, 1.0);

    //delete base
    if (base != NULL)
    {
        for (int d = 0; d < dim; ++d)
        {
            if (base[d].direction != NULL)
            {
                delete[] base[d].direction;
                base[d].direction = NULL;
            }
        }
        delete[] base;
        base = NULL;
    }

    // entory data points into hash table
    storePoint();

    if (convertData != NULL)
    {
        for (int n = 0; n < length; n++)
        {
            if (convertData[n] != NULL)
            {
                delete[] convertData[n];
                convertData[n] = NULL;
            }
        }
        delete[] convertData;
        convertData = NULL;
    }
}

void Index::Build(int _dim, unsigned num, void** data)
{
    dim = _dim;
    //Principal Component Analysis
    cout << "calculate PCA ." << endl;
    PrincipalComponentAnalysis pca;

    pca.executePCA(dim, num, (featureElement**)data);
    originalData = Mat(num, dim, CV_8UC1);
    for (size_t y = 0; y < num; y++)
    {
        memcpy(originalData.data + y * originalData.step, (featureElement*)(data[y]), sizeof(featureElement) * dim);
    }

    // copy PCA direction to base_t for BDH
    const PC_t* pcDir = pca.getPCdir();
    base_t* base = new base_t[dim];
    for (int d = 0; d < dim; ++d)
    {
        base[d].mean = pcDir[d].mean;           // pca.mean
        base[d].variance = pcDir[d].variance;   // pca.eigenvalues
        base[d].direction = new double[dim];    // pca.eigenvectors
        memcpy(base[d].direction, pcDir[d].direction, sizeof(double)*dim);
    }

    cout << "training Start ." << endl;
    // train parameters
    parameterTuning_ICCV2013(dim, num, data, base, P, 13, M, hashSize, pointSize, entrySize, variance, hashTable, delta, subspace, lestspace, 0.1, 1.0);

    //delete base
    for (int d = 0; d < dim; ++d)
    {
        delete[] base[d].direction;
    }
    delete[] base;

    // entory data points into hash table
    storePoint();
}


Index::Index(int _dim, unsigned num, void** data)
    : dim(_dim)
    , M(0)
    , P(10)
    , bit(0)
    , delta(0.0)
    , pointSize(0)
    , entrySize(0)
    , hashSize(0)
    , hashTable()
{
    Build(dim, num, data);
}

bool Index::loadTable(const String & path)
{
    return hashTable.readTable(path);
}

bool Index::saveTable(const String & path) const
{
    return hashTable.writeTable(path);
}

bool Index::loadParameters(const String& path)
{

    ifstream ifs(path);
    if (ifs.is_open() == false)
    {
        return false;
    }

    ifs >> dim
        >> M
        >> P
        >> bit
        >> delta
        >> pointSize
        >> entrySize
        >> hashSize
        >> variance;

    Subspace::dim = dim;

    subHashSizeMax = 0;

    for (int m = 0; m < M; ++m)
    {
        Subspace stubSpace;
        stubSpace.subDim = P;

        ifs >> stubSpace.subHashSize
            >> stubSpace.variance;

        if (stubSpace.subHashSize > subHashSizeMax)
        {
            subHashSizeMax = stubSpace.subHashSize;
        }

        for (int sd = 0; sd < P; ++sd)
        {
            Mat stub(1, dim, CV_64FC1);
            for (int d = 0; d < dim; ++d)
            {
                double v;
                ifs >> v;
                stub.at<double>(0, d) = v;
            }
            stubSpace.baseVector.push_back(stub);
        }

        for (int i = 0; i < stubSpace.subHashSize; ++i)
        {
            size_t stubHash;
            double c;
            ifs >> c >> stubHash;
            stubSpace.hashKey.push_back(stubHash);
            stubSpace.cellVariance.push_back(c);

            std::vector<double> stub;
            for (int d = 0; d < P; ++d)
            {
                double v;
                ifs >> v;
                stub.push_back(v);
            }
            stubSpace.centroidVector.push_back(stub);
        }
        subspace.push_back(stubSpace);
    }

    ifs >> lestspace.subDim
        >> lestspace.variance;

    std::vector<double> stubDouble;
    for (int sd = 0; sd < lestspace.subDim; ++sd)
    {
        double v;
        ifs >> v;
        stubDouble.push_back(v);
    }
    lestspace.centroidVector.push_back(stubDouble);
    lestspace.cellVariance.push_back(lestspace.variance);

    for (int sd = 0; sd < lestspace.subDim; ++sd)
    {
        Mat stub(1, dim, CV_64FC1);
        for (int i = 0; i < dim; i++)
        {
            double v;
            ifs >> v;
            stub.at<double>(0, i) = v;
        }
        lestspace.baseVector.push_back(stub);
    }

    ifs.close();

    hashTable.initialize(entrySize, hashSize);
    return true;
}


bool Index::saveParameters(const String& path) const
{

    ofstream ofs(path);
    if (ofs.is_open() == false)
    {
        return false;
    }

    ofs << dim << "\t"
        << M << "\t"
        << P << "\t"
        << bit << "\t"
        << delta << "\t"
        << pointSize << "\t"
        << entrySize << "\t"
        << hashSize << "\t"
        << variance << endl;

    for (int m = 0; m < M; ++m)
    {
        ofs << subspace[m].subHashSize << "\t"
            << subspace[m].variance << endl;

        for (int sd = 0; sd < P; ++sd)
        {
            for (int d = 0; d < dim; ++d)
            {
                ofs << subspace[m].baseVector.at<double>(sd, d) << "\t";
            }
            ofs << endl;
        }

        for (int i = 0; i < subspace[m].subHashSize; ++i)
        {

            ofs << subspace[m].cellVariance[i] << "\t" << subspace[m].hashKey[i] << endl;

            for (int d = 0; d < subspace[m].subDim; ++d)
            {
                ofs << subspace[m].centroidVector[i][d] << "\t";
            }

            ofs << endl;
        }
    }

    ofs << lestspace.subDim << "\t"
        << lestspace.variance << endl;
    for (int sd = 0; sd < lestspace.subDim; ++sd)
    {
        ofs << lestspace.centroidVector[0][sd] << "\t";
    }
    ofs << endl;

    for (int sd = 0; sd < lestspace.subDim; ++sd)
    {
        for (int d = 0; d < dim; ++d)
        {
            ofs << lestspace.baseVector.at<double>(sd, d) << "\t";
        }
        ofs << endl;
    }


    ofs.close();
    return true;
}

void Index::storePoint(/*index_t num, data_t** data*/)
{
    //alloc workspace
    collision_t* collision = new collision_t[hashSize];
    memset(collision, 0, sizeof(collision_t)*hashSize);

    size_t* hashKey = new size_t[originalData.rows];
    for (int n = 0; n < originalData.rows; n++)
    {
        //get hash value
        hashKey[n] = hashFunction(n);
        //increment collision
        ++collision[hashKey[n]];
    }

    hashTable.allocTable(collision);
    delete[] collision;

    char* entry = new char[entrySize];
    if (entry == nullptr)
    {
        exit(__LINE__);
    }

    for (int n = 0; n < originalData.rows; ++n)
    {
        memcpy(entry, originalData.row(n).data, pointSize);
        *reinterpret_cast<index_t*>(entry + pointSize) = n;

        hashTable.storeEntryWithoutAlloc(hashKey[n], entry);
    }
    delete[] entry;

    delete[] hashKey;
}

template<typename data_t>
double innerProduct(const std::vector<double>& base, const data_t* data)
{
    double val = 0.0;
    for (size_t i = 0; i < base.size(); i++)
    {
        val += base[i] * data[i];
    }
    return val;
}

template<typename base_t, typename data_t>
double innerProduct(const base_t *base, const data_t* data, size_t length)
{
    double val = 0.0;
    for (size_t i = 0; i < length; i++)
    {
        val += base[i] * data[i];
    }
    return val;
}

#define INNER_PRODUCT(index,dst,src1,src2,type2) for(index = 0;index < src1.rows;index++) { dst[index] = innerProduct((double*)src1.row(index).data, (type2*)src2.data, src1.cols); }

void Subspace::getPCAdata(const Mat &data, double* PCAdata) const
{
    int i = 0;
    switch (data.depth())
    {
    case CV_8U:  INNER_PRODUCT(i, PCAdata, baseVector, data, unsigned char); break;
    case CV_8S:  INNER_PRODUCT(i, PCAdata, baseVector, data, char); break;
    case CV_16U: INNER_PRODUCT(i, PCAdata, baseVector, data, unsigned short); break;
    case CV_16S: INNER_PRODUCT(i, PCAdata, baseVector, data, short); break;
    case CV_32S: INNER_PRODUCT(i, PCAdata, baseVector, data, int); break;
    case CV_32F: INNER_PRODUCT(i, PCAdata, baseVector, data, float); break;
    case CV_64F: INNER_PRODUCT(i, PCAdata, baseVector, data, double); break;
    default:
        CV_Error(Error::StsUnsupportedFormat, "unsupported type");
        break;
    }
}

#undef INNER_PRODUCT

void cv::bdh::Subspace::getPCAdata(const Mat & data, Mat & PCAdata) const
{
    CV_Assert(data.size() == PCAdata.size());
    CV_Assert(data.type() == PCAdata.type());
    Mat stub = PCAdata.reshape(1, 1);
    for (int i = 0; i < baseVector.rows; i++)
    {
        stub.col(i) = baseVector.row(i).dot(data);
    }
}

size_t Index::hashFunction(int index)
{

    size_t hashKey = 0;
    for (int m = 0; m < M; ++m)
    {
        hashKey += subspace[m].getSubHashValue(originalData.row(index));
    }
    return hashKey;
}


int Index::getBucketList(
    InputArray _query,
    double searchParam,
    search_mode searchMode,
    std::vector<hashKey_t>& bucketList
    )const
{
    Mat query = _query.getMat();

    //部分距離を計算し，優先度の高い順にソート
    layer_t* layer = new layer_t[M];
    for (int m = 0; m < M; ++m)
    {
        layer[m].node = new node_t[subspace[m].subHashSize + 1];
        layer[m].node[subspace[m].subHashSize].distance = DBL_MAX;
    }
    setLayerParam(layer, query);

    unsigned NNC = 0;
    status_t status;
    switch (searchMode)
    {
    case Radius:
    {
        //ハッシュに使っていない基底における重心からの距離を求める
        //NumPointsアルゴリズムの場合、バケットの選択に影響しない計算。
        //頂点数に対して次元数が大きいほど、冗長になる
        double* lestSpaceVal = new double[lestspace.dim];
        lestspace.getPCAdata(query, lestSpaceVal);
        delete[] lestSpaceVal;

        //Radius以下の距離を探索
        NNC = NearBucket_R(searchParam, layer, status, bucketList);

        break;
    }
    case NumPoints:
    {
        unsigned C = static_cast<unsigned>(searchParam);
        bucketList.reserve(C);

        for (double Lbound = 0, Ubound = layer[0].restMin + 1.0e-10
            ; NNC < C
            ; Ubound += delta)
        {
            NNC += NearBucket_C(Lbound, Ubound, layer, status, bucketList);
            Lbound = Ubound;
        }

        break;
    }

    case NumPoints2:
    {
        unsigned C = static_cast<unsigned>(searchParam);
        bucketList.reserve(C);

        //前回の探索で打ち切られたルートを再探索
        list<status_t> statusQue;
        statusQue.push_front(status);//ルートノード
        list<status_t>::iterator itr;
        for (double Rbound = layer[0].restMin + 1.0e-10
            ; NNC < C
            ; Rbound += delta)
        {
            size_t loop = statusQue.size();
            itr = statusQue.begin();
            for (size_t l = 0; l < loop; ++l)
            {
                NNC += NearBucket_C_list(Rbound, layer, statusQue, &itr, bucketList);
            }
        }
        break;
    }
    }

    //探索が終わったのでデリート
    for (int m = 0; m < M; ++m)
    {
        delete[] layer[m].node;
    }
    delete[] layer;

    return NNC;
}
} // bdh
} // cv
