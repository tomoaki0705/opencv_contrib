#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/bdh.hpp"
#include "opencvPCA.h"
#include "Subspace.h"
#include <BDHtraining.h>
#include <BDH.h>
const double deltaRate = 50;

///************** Indexing *******************/
namespace cv {
namespace bdh {
template<typename data_t>
void cv::bdh::Subspace::setNodeParam(node_t * node, data_t * query)
{

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

template <typename data_t>
void Index<data_t>::setLayerParam(
    layer_t* layer,
    data_t* query
) const {

    layer_t* layer_p = layer;
    layer_t* layer_p_end = layer + M;
    for (int m = 0; layer_p != layer_p_end; ++m, ++layer_p)
    {
        layer_p->k = subspace[m].subHashSize;
        subspace[m].setNodeParam(layer_p->node, query);
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

template <typename data_t>
int Index<data_t>::NearBucket_R(
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

template <typename data_t>
int Index<data_t>::NearBucket_C(
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

template <typename data_t>
int Index<data_t>::NearBucket_C_list(
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
int Index<data_t>::searchInBucket(
    data_t* query,
    size_t hashKey,
    std::priority_queue<point_t>& NNpointQue
) const {

    bin_t bin;
    hashTable.getBin(hashKey, bin);

    /*set point's pointer*/
    collision_t coll = bin.collision;
    address_t addr = bin.addressOfChainList;
    address_t addr_end = addr + coll*entrySize;
    double KNNdist = NNpointQue.top().distance;

    double dist;
    data_t* data;
    data_t* query_p;
    data_t* query_end = query + dim;
    while (addr != addr_end)
    {
        /*Distance Calculation*/
        query_p = query;
        data = reinterpret_cast<data_t*>(addr);
        dist = 0.0;
        while ((query_p != query_end) && (dist < KNNdist))
        {
            dist += NORM((*query_p++) - (*data++));
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
template <typename data_t>
void Index<data_t>::linearSearchInNNcandidates(
    data_t* query,
    point_t* point,
    int K,
    double epsilon,
    std::vector<hashKey_t>& bucketList
) const
{
    //生成されたハッシュキーを元にバケットを参照して最近傍点を探索 start//
    priority_queue<point_t> NNpointQue;
    //最近傍点保持用のヒープ木を初期化
    for (int i = 0; i < K; ++i)
    {
        NNpointQue.push(point_t(-1, epsilon));
    }

    //見つけてきたハッシュキーを参照して最近傍点を探索する
    vector<hashKey_t>::iterator keyList_itr = bucketList.begin();
    vector<hashKey_t>::iterator keyList_itr_end = bucketList.end();
    for (; keyList_itr != keyList_itr_end; ++keyList_itr)
    {
        searchInBucket(query, (*keyList_itr).hashKey, NNpointQue);
    }
    //生成されたハッシュキーを元にバケットを参照して最近傍点を探索 end//

    //優先度付きキュー内の最近傍点を返却用引数にコピー
    for (int i = K - 1; i >= 0; --i)
    {
        point[i] = NNpointQue.top();
        NNpointQue.pop();
    }
}


template <typename data_t>
int Index<data_t>::NearestNeighbor(
    data_t* query,
    point_t* point,
    double searchParam,
    search_mode searchMode,
    int K,
    double epsilon
)const
{

    // Generate the Bucket Hash key of the needle (begin)//
    vector<hashKey_t> bucketList;
    int NNC = getBucketList(query, searchParam, searchMode, bucketList);

    // Generate the Bucket Hash key of the needle (end)//
    linearSearchInNNcandidates(query, point, K, epsilon, bucketList);

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

bool readBinary(const String &path, unsigned &dim, unsigned &num, featureElement ** & data)
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
    data = new featureElement*[num];

    const int dSize = sizeof(featureElement)*dim;
    for (unsigned n = 0; n < num; n++) {
        data[n] = new featureElement[dim];
        ifs.read((char*)data[n], dSize);
    }
    ifs.close();

    return true;
}

template <typename data_t>
void parameterTuning(const Mat& originalData, const PCA& featureSpace, enum tuning_method method = cv::bdh::TUNING_ADVANCED_2013)
{
    switch (method)
    {
    case cv::bdh::TUNING_ORIGINAL:
        break;
    case cv::bdh::TUNING_ADVANCED_2013:
    default:
        parameterTuning_ICCV2013(originalData.cols, originalData.rows, data, base, P, 13, M, hashSize, pointSize, entrySize, variance, hashTable, delta, subspace, lestspace, 0.1, 1.0);
        break;
    }
}

template <typename data_t>
void parameterTuning_ICCV2013(int dim, index_t num, data_t ** const data, base_t * const base, int P, int bit, int &M, size_t &hashSize, size_t &pointSize, size_t &entrySize, double &variance, HashTable &hashTable, double &delta, Subspace* &subspace, Subspace& lestspace, double bit_step = 1.0, double sampling_rate = 1.0)
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

    subspace = new Subspace[M];
    size_t rank = 1;

    for (int m = 0; m < M; ++m)
    {
        subspace[m].setParameters(baseSet[m]);

        subspace[m].hashKey = new size_t[subspace[m].subHashSize];

        for (int i = 0; i < subspace[m].subHashSize; ++i)
        {
            subspace[m].hashKey[i] = rank*i;

        }

        rank *= subspace[m].subHashSize;
    }

    lestspace.subDim = lestSet.subDim;
    lestspace.variance = lestSet.variance;  // pca.eigenvalues
    lestspace.centroid = new double*[1];
    lestspace.centroid[0] = new double[lestSet.subDim];

    lestspace.base = new double*[lestspace.subDim];
    for (int d = 0; d < lestspace.subDim; ++d)
    {
        // pca.mean
        lestspace.centroid[0][d] = lestSet.base[d].mean;
        // pca.eigenvectors
        lestspace.base[d] = new double[dim];
        memcpy(lestspace.base[d], lestSet.base[d].direction, sizeof(double)*dim);
    }

}

template<typename data_t>
void cv::bdh::Index<data_t>::Build(InputArray data, PCA::Flags order)
{
    cv::Mat _data = data.getMat();
    originalData = _data.clone();
    cv::PCA pca(originalData, Mat(), order, order == PCA::DATA_AS_ROW ? originalData.rows : originalData.cols);
    pca.mean;
    pca.eigenvectors;
    pca.eigenvalues;
    int i = 0;
    int length = originalData.rows;
    dim = originalData.cols;

    // copy PCA direction to base_t for BDH
    base_t* base = new base_t[dim];
    for (int d = 0; d < dim; ++d)
    {
        base[d].mean = (double)pca.mean.at<float>(d);
        base[d].variance = pca.eigenvalues.at<float>(d);
        base[d].direction = new double[dim];    // pca.eigenvectors
        for (size_t x = 0; x < dim; x++)
        {
            base[d].direction[x] = (double)((float*)(pca.eigenvectors.data + d * pca.eigenvectors.step))[x];
        }
    }
    featureElement **convertData = new featureElement*[length];
    for (size_t n = 0; n < length; n++)
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
        for (size_t n = 0; n < length; n++)
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

template <typename data_t>
void Index<data_t>::Build(int dim, unsigned num, void** data)
{
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

template <typename data_t>
Index<data_t>::Index(int dim, unsigned num, void** data)
    : dim(dim)
    , M(0)
    , P(10)
    , bit(0)
    , delta(0.0)
    , pointSize(0)
    , entrySize(0)
    , hashSize(0)
    , subspace(nullptr)
    , hashTable()
{
    Build(dim, num, data);
}

template <typename data_t>
bool Index<data_t>::loadParameters(
    const String& path
)
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
    subspace = new Subspace[M];
    for (int m = 0; m < M; ++m) {
        subspace[m].subDim = P;

        ifs >> subspace[m].subHashSize
            >> subspace[m].variance;

        if (subspace[m].subHashSize > subHashSizeMax)
        {
            subHashSizeMax = subspace[m].subHashSize;
        }

        subspace[m].base = new double*[P];
        for (int sd = 0; sd < P; ++sd)
        {
            subspace[m].base[sd] = new double[dim];
            for (int d = 0; d < dim; ++d)
            {
                ifs >> subspace[m].base[sd][d];
            }
        }

        subspace[m].cellVariance = new double[subspace[m].subHashSize];
        subspace[m].hashKey = new size_t[subspace[m].subHashSize];
        subspace[m].centroid = new double*[subspace[m].subHashSize];
        for (int i = 0; i < subspace[m].subHashSize; ++i)
        {
            ifs >> subspace[m].cellVariance[i] >> subspace[m].hashKey[i];

            subspace[m].centroid[i] = new double[P];
            for (int d = 0; d < P; ++d)
            {
                ifs >> subspace[m].centroid[i][d];
            }
        }
    }

    ifs >> lestspace.subDim
        >> lestspace.variance;

    lestspace.centroid = new double*[1];
    lestspace.centroid[0] = new double[lestspace.subDim];
    for (int sd = 0; sd < lestspace.subDim; ++sd)
    {
        ifs >> lestspace.centroid[0][sd];
    }
    lestspace.cellVariance = new double[1];
    lestspace.cellVariance[0] = lestspace.variance;

    lestspace.base = new double*[lestspace.subDim];
    for (int sd = 0; sd < lestspace.subDim; ++sd)
    {
        lestspace.base[sd] = new double[dim];
        for (int d = 0; d < dim; ++d) {
            ifs >> lestspace.base[sd][d];
        }
    }

    ifs.close();

    hashTable.initialize(entrySize, hashSize);
    return true;
}

template <typename data_t>
bool Index<data_t>::saveParameters(const String& path) const
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
                ofs << subspace[m].base[sd][d] << "\t";
            }
            ofs << endl;
        }

        for (int i = 0; i < subspace[m].subHashSize; ++i)
        {

            ofs << subspace[m].cellVariance[i] << "\t" << subspace[m].hashKey[i] << endl;

            for (int d = 0; d < subspace[m].subDim; ++d)
            {
                ofs << subspace[m].centroid[i][d] << "\t";
            }

            ofs << endl;
        }
    }

    ofs << lestspace.subDim << "\t"
        << lestspace.variance << endl;
    for (int sd = 0; sd < lestspace.subDim; ++sd)
    {
        ofs << lestspace.centroid[0][sd] << "\t";
    }
    ofs << endl;

    for (int sd = 0; sd < lestspace.subDim; ++sd)
    {
        for (int d = 0; d < dim; ++d)
        {
            ofs << lestspace.base[sd][d] << "\t";
        }
        ofs << endl;
    }


    ofs.close();
    return true;
}

template <typename data_t>
void Index<data_t>::storePoint(/*index_t num, data_t** data*/)
{
	//alloc workspace
	collision_t* collision = new collision_t[hashSize];
	memset(collision, 0, sizeof(collision_t)*hashSize);

	size_t* hashKey = new size_t[originalData.rows];
	for (index_t n = 0; n < originalData.rows; ++n)
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

	for (index_t n = 0; n < originalData.rows; ++n)
	{
		memcpy(entry, originalData.row(n).data, pointSize);
		*reinterpret_cast<index_t*>(entry + pointSize) = n;

		hashTable.storeEntryWithoutAlloc(hashKey[n], entry);
	}
	delete[] entry;

	delete[] hashKey;
}

template<typename data_t>
double Subspace::innerProduct(double* base, const data_t* data) const
{
	double val = 0.0;
	for (int d = 0; d < dim; ++d){
		val += base[d] * data[d];
	}
	return val;
}

template<typename data_t>
void Subspace::getPCAdata(const data_t* data, double* PCAdata) const
{
	double** base_p = base;
	double* PCAdata_end = PCAdata + subDim;
	for (; PCAdata != PCAdata_end; ++PCAdata)
	{
		*PCAdata = innerProduct(*base_p++, data);
	}
}

template<typename data_t>
size_t Subspace::getSubHashValue(
	const data_t* data
	) const
{

	//work space
	double* PCAdata = new double[subDim];

	getPCAdata(data, PCAdata);
	int idx = NearestNeighbor(subDim, subHashSize, centroid, PCAdata);
	delete[] PCAdata;

	return hashKey[idx];
}

template <typename data_t>
size_t Index<data_t>::hashFunction(int index)
{

	size_t hashKey = 0;
	for (int m = 0; m < M; ++m)
	{
		hashKey += subspace[m].getSubHashValue(originalData.row(index).data);
	}
	return hashKey;
}

template <typename data_t>
int Index<data_t>::getBucketList(
	data_t* query,
	double searchParam,
	search_mode searchMode,
	std::vector<hashKey_t>& bucketList
	)const
{

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
		const double lestSpaceDist = lestspace.getDistanceToCentroid(lestSpaceVal, 0);
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