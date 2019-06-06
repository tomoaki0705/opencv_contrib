#ifndef __OPENCV_BDH_HPP__
#define __OPENCV_BDH_HPP__

#include <opencv2/core.hpp>
#include <queue>
#include <list>
/**
* @defgroup bdh Bucket Distance Hash
* This module computes the nearest neighor distance
*/
namespace cv {
namespace bdh {
    typedef unsigned char featureElement;
    /////////////// Load Function ////////////////////////
    /**
    * @brief read point set
    * @return is file open ?
    * @param path path of binary file
    * @param dim  dimension of features
    * @param num  number of features
    * @param data actual output
    */
    CV_EXPORTS bool readBinary(const String &path, unsigned &dim, unsigned &num, OutputArray data, int type = CV_8UC1);
    CV_EXPORTS bool readCorrectClass(const String& filename, std::vector<int>& correctClass);

    typedef unsigned collision_t;//!< type of collision
    typedef char* address_t;     //!< type of address

                                 /**
                                 * @brief a chain list
                                 */
    struct bin_t
    {
        collision_t collision;       //!< collision
        address_t addressOfChainList;//!< head address of chain list
    };

    enum search_mode
    {
        Radius
       ,NumPoints
       ,NumPoints2
    };
    enum tuning_method
    {
        TUNING_ORIGINAL
       ,TUNING_ADVANCED_2013
    };

    /**
    * @brief type of node. a node coresponde to a centroid
    */
    struct node_t {
        size_t hashKey;//!< hash value
        double distance; //!< sub bucket distance

                         /**
                         * @brief compare the distance
                         * @return is which lesser ?
                         */
        bool operator < (
            const node_t& obj //!< compared object
            )
        {
            return distance < obj.distance;
        }
    };

    /**
    * @brief type of layer. a node coresponde to a subspace
    */
    struct layer_t
    {
        int k;          //!< number of nodes
        double restMin; //!< the minimam rest distance from this layer
        double restMax; //!< the maximam rest distance from this layer
        double gap;     //!< the gap of distance between max and min
        node_t* node;   //!< nodes.

        void calc_gap()
        {
            gap = node[k - 1].distance - node[0].distance;
        }

        /**
        * @brief compare the gap
        * @return is which gap larger ?
        */
        bool operator < (const layer_t& obj)
        {
            return gap > obj.gap;
        }
    };

    //* @brief status of neare bucket search
    //*/
    struct status_t
    {
    
        int m;          //!< index of layer
        int nodeIdx;    //!< index of nodes
        double dist = 0.0;//!< distance
        size_t hashKey; //!< hash value
    
        /**
        * @brief default constructor
        */
        status_t()
            : m(0)
            , nodeIdx(0)
            , dist(0.0)
            , hashKey(0)
        {}
    
        /**
        * @brief constructor
        */
        status_t(
            const int& _m,           //!< index of layer
            const int& _nodeIdx,     //!< index of nodes
            const size_t& _hashKey,  //!< hash value
            const double& _dist)     //!< distance
            : m(_m)
            , nodeIdx(_nodeIdx)
            , dist(_dist)
            , hashKey(_hashKey)
        {}
    
        /**
        * @brief constructor
        */
        status_t(
            const int& _m            //!< index of layer
            )
            : m(_m)
            , nodeIdx(0)
            , dist(0.0)
            , hashKey(0)
        {}
    };
    
    /**
    * @brief status of neare bucket search
    */
    struct hashKey_t {

        size_t hashKey; //!< hash value
        double dist;    //!< distance

                        /**
                        * @brief default constructor
                        */
        hashKey_t()
            :hashKey(0)
            , dist(0)
        {}

        /**
        * @brief constructor. allocate memory of nodeIdx and deep copied.
        */
        hashKey_t(
            size_t _hashKey, //!< hash value
            double _dist)    //!< distance
            : hashKey(_hashKey)
            , dist(_dist)
        {}

        void setVariable(size_t _hashKey, double _dist)
        {
            hashKey = _hashKey;
            dist = _dist;
        }
    };

    /**
    * @struct point_t
    * @brief this structure has propaties of a point
    */
    struct point_t
    {

        size_t index;           //!< index of data
        double distance;        //!< ditance from query

        /**
        * @brief default constructor
        */
        point_t()
        {}

        /**
        * @brief constructor
        */
        point_t(
            const size_t& _index,    //!< [in] the index of point 
            const double& _distance  //!< [in] the distance from query
        )
            : index(_index)
            , distance(_distance)
        {}

        /**
        * @brief constructor
        */
        point_t(
            const size_t& _index     //!< [in] the index of point 
        )
            : index(_index)
        {}

        /**
        * @brief set member variables
        */
        void setMemberVariable(
            const size_t& _index,   //!< [in] the index of point
            const double& _distance //!< [in] the distance from query
        )
        {
            index = _index;
            distance = _distance;
        }

        /**
        * @brief compare the distance
        * @return is my distance lessor than e's ?
        */
        bool operator <(
            const point_t &e    //!< [in] compare object
            ) const
        {
            return distance < e.distance;
        }

        /**
        * @brief compare the distance
        * @return is my distance equal to e's ?
        */
        bool operator ==(
            const point_t &e    //!< [in] compare object
            ) const
        {
            return index == e.index;
        }
    };
    /**
    * @brief hash table
    */
    class HashTable
    {

    private:
        static const size_t collisionSize = sizeof(collision_t); //!< bytes of collision

        size_t entrySize;    //!< bytes of entry
        size_t hashSize;     //!< size of hash table
        size_t nEntry;
        address_t* hashTable;//!< hash table

    public:

        /**
        * @brief default constructor
        */
        HashTable()
            : entrySize(0)
            , hashSize(0)
            , nEntry(0)
            , hashTable(nullptr)
        {}

        size_t get_nEntry() const
        {
            return nEntry;
        }

        /**
        * @brief constructor
        */
        HashTable(size_t _entrySize, size_t _hashSize)
            : entrySize(_entrySize)
            , hashSize(_hashSize)
        {
            initialize(entrySize, hashSize);
        }

        /**
        * @brief destructor
        */
        ~HashTable()
        {
            if (hashTable == nullptr)
            {
                return;
            }

            address_t* table_p = hashTable;
            for (size_t hv = 0; hv < hashSize; ++hv)
            {
                if (*table_p != nullptr)
                {
                    free(*table_p);
                }
                ++table_p;
            }

            delete[] hashTable;
            hashTable = nullptr;
        }

        /**
        * @brief initializer
        */
        void initialize(
            size_t pointSize,//!< [in] sizeof(data_t)*dim;
            size_t hashSize  //!< [in] hash size = 1<<bit;
        );

        /**
        * @brief bin getter
        */
        void getBin(size_t hashKey, bin_t& bin) const
        {
            address_t table_p = hashTable[hashKey];
            if (!table_p)
            {
                bin.collision = 0;
                return;
            }
            bin.collision = *reinterpret_cast<collision_t*>(table_p);
            bin.addressOfChainList = table_p + collisionSize;
        }


        /**
        * @brief collision getter
        */
        collision_t getCollision(
            size_t hashKey //!< [in] hash value
        ) const
        {
            address_t table_p = hashTable[hashKey];
            if (table_p)
            {
                return (*reinterpret_cast<collision_t*>(table_p));
            }
            else
            {
                return 0;
            }
        }

        /**
        * @brief check is this hash value
        */
        address_t getPointer(
            size_t hashKey//!< [in] hash value
        ) const
        {
            return hashTable[hashKey];
        }

        /**
        * @brief check is hash value available
        * @return is hashTable[hashKey] active ?
        */
        bool isEntried(
            size_t hashKey //!< [in] hash vlue
        )const
        {
            if (hashTable[hashKey])
            {
                return true;
            }
            else
            {
                return false;
            }
        };


        /**
        * @brief Storing a point into hash table
        * @return is allocation complete ?
        */
        bool storeEntry(
            size_t HashValue,     //!< [in] hash value of point data
            const address_t point//!< [in] the point stored into hash table
        );

        /**
        * @brief allocation of hash table
        * @return is allocation complete ?
        */
        bool allocTable(
            unsigned* collision//!< [in] collision list of the hash table
        );

        /**
        * @brief you have to take care of whether allocation of hash table has already completed
        */
        void storeEntryWithoutAlloc(
            size_t HashValue,    //!< [in] hash value of point data
            const address_t data//!< [in] the point stored into hash table 
        );

        /**
        * @brief read hash table from binary file
        * @return is file open ?
        */
        bool readTable(
            const String& tblFile//!< [in] file path
        );

        /**
        * @brief write hash table from binary file
        * @return is file open ?
        */
        bool writeTable(
            const String& tblFile//!< [in] file path
        ) const;

    };

    struct base_t
    {
        static int dim;     //!< dimension of data space (static member)

        int idx;            //!< index of base
        double mean;        //!< pca.mean           mean at base direction
        double variance;    //!< pca.eigenvalues    variance at base direction
        double* direction;  //!< pca.eigenvectors   direction of base [dim]

                            /**
                            * @brief default constructor
                            */
        base_t()
            : idx(0)
            , mean(0.0)
            , variance(0.0)
            , direction(nullptr)
        {}

        /**
        * @brief initialize all attributes default and release memory
        */
        void clear()
        {
            idx = 0;
            mean = 0.0;
            variance = 0.0;

            if (direction != nullptr)
            {
                delete[] direction;
            }

        }

        /**
        * @brief compare the variance
        */
        bool operator < (
            const base_t& obj //!< [in] object
            )
        {
            return variance > obj.variance;
        }

    };

    /**
    * @brief set of base_t
    * @details destructor and copy constructor are not defined for fast sort
    */
    struct baseset_t {

        int idx;               //!< index of base set
        int subDim;            //!< number of base
        double variance;       //!< sum of variance of base

        base_t* base;          //!< set of base_t[subDim]

        int k;                 //!< number of centorid = 2^bit
        double bit;            //!< amount of infomation for this base set = log2(k)
        double error;          //!< error of quantization = sum of cellVariance
        double score;          //!< efficiency for incrementint the number of centroid
        double** centroid;     //!< centroid[k][subDim]
        double* cellVariance;  //!< variance in cell[k]

                                /**
                                * @brief default constructor
                                */
        baseset_t()
            : idx(0)
            , subDim(0)
            , variance(0.0)
            , base(nullptr)
            , k(0)
            , bit(0)
            , error(0.0)
            , score(0.0)
            , centroid(nullptr)
            , cellVariance(nullptr)
        {}

        /**
        * @brief initialize all attributes default and release memory
        */
        void clear()
        {
            for (int sd = 0; sd < subDim; ++sd)
            {
                base[sd].clear();
            }

            if (centroid != nullptr)
            {
                for (int i = 0; i < k; ++i)
                {
                    delete[] centroid[i];
                }
                delete[] centroid;
                centroid = nullptr;
            }

            if (cellVariance != nullptr)
            {
                delete[] cellVariance;
                cellVariance = nullptr;
            }

        }

        /**
        * @brief compare the variance
        */
        bool operator < (
            const baseset_t& obj //!< [in] object
            )
        {
            return variance > obj.variance;
        }

    };

    class CV_EXPORTS Subspace
    {

    public:
        static int dim;      //!< dimension
        int subDim;          //!< dimension of subspace
        int subHashSize;     //!< hash size at subspace = 1<<bit
        double bit;          //!< information volume
        double variance;     //!< sum of variance
        Mat baseVector;      //!< base direction[P][dim]
        std::vector<size_t> hashKey;                      //!< hash value of bin corespond to centroid[subHashSize]
        std::vector<double> cellVariance;                 //!< variance in cell[subHashSize]
        std::vector<std::vector<double> > centroidVector; //!< centroid[subHashSize][subDim]

    public:

        /**
        * @brief default constructor
        */
        Subspace()
            : subDim(0)
            , subHashSize(0)
            , bit(0.0)
            , variance(0.0)
        {}

        ~Subspace()
        {
        }

        /**
        * @brief set training paramters
        */
        void setParameters(
            const baseset_t& baseSet
        );

        /**
        * @brief project data into Principal Component space
        */
        void getPCAdata(const Mat & data, double * PCAdata) const;

        /**
        * @brief project data into Principal Component space
        */
        void getPCAdata(const Mat & data, Mat & PCAdata) const;

        /**
        * @brief get sub hash value
        */
        size_t getSubHashValue(InputArray _data) const;

        /**
        * @brief set node param
        */
        void setNodeParam(
            node_t* node,
            InputArray _query
        ) const;

        /**
        * @brief get distance to centroid
        */
        double getDistanceToCentroid(
            double* PCAquery,
            int centroidIndex
        )const;
    };

    typedef unsigned index_t;//!< type of index for point

    class CV_EXPORTS Index
    {
    private:

        int dim;            //!< dimension of dataspace
        int M;              //!< number of subspace
        int P;              //!< dimension of subspace
        int bit;            //!< bits num of hash table
        double delta;       //!< increment step of search radius for C search
        int subHashSizeMax; //!< max of sub hash size
        size_t pointSize;   //!< number of data points
        size_t entrySize;   //!< size of entry = sum of size of index and data point
        size_t hashSize;    //!< hash size = 2^bit
        double variance;
        Mat originalData;   //!< store the original feature data

        std::vector<Subspace> subspace;//!< classes handling parameters of subspace
        Subspace  lestspace;//!< classe handling parameters of subspace not which construct the hash table

        HashTable hashTable;    //!< hash table

    public:

        /**
        * @brief default constructor
        */
        Index()
            : dim(0)
            , M(0)
            , P(10)
            , bit(0)
            , delta(0.0)
            , pointSize(0)
            , entrySize(0)
            , hashSize(0)
            , hashTable()
        {}

        Index(int _dim, unsigned num, void** data);

        void Build(int _dim, unsigned num, void** data);
        void Build(InputArray data, enum PCA::Flags order = PCA::DATA_AS_ROW);

        ~Index()
        {
        }

        size_t get_nDdataPoints() const
        {
            return hashTable.get_nEntry();
        }

        double get_variance() const
        {
            return variance;
        }

        bool loadTable(const String & path);
        bool saveTable(const String & path) const;
        bool loadParameters(const String & path);
        bool saveParameters(const String & path) const;

        ///////////// Search Function ////////////////////////

        void setLayerParam(layer_t * layer, InputArray _query) const;

        int NearBucket_R(const double Radius, layer_t * const layer, const status_t & status, std::vector<hashKey_t>& bucketList) const;

        int NearBucket_C(const double & Lbound, const double & Ubound, layer_t * const layer, const status_t & status, std::vector<hashKey_t>& bucketList) const;

        int NearBucket_C_list(const double Rbound, layer_t * const layer, std::list<status_t>& statusQue, std::list<status_t>::iterator * itr, std::vector<hashKey_t>& bucketList) const;

        int searchInBucket(InputArray _query, size_t hashKey, std::priority_queue<point_t>& NNpointQue) const;

        void linearSearchInNNcandidates(InputArray _query, point_t* point, int K, double epsilon, std::vector<hashKey_t>& bucketList) const;

        /**
        * @brief search in Bucket Distance R from query
        * @return number of points in search area
        */
        int NearestNeighbor(
            InputArray _query,
            point_t* point,
            double searchParam,
            search_mode searchMode = NumPoints,
            int K = 1,
            double epsilon = DBL_MAX
        )const;

        int getBucketList(
            InputArray _query,
            double searchParam,
            search_mode searchMode,
            std::vector<hashKey_t>& bucketList
        )const;

        ////////// store data points /////////////////

        /**
        * @brief store point set into hash table
        */
        void storePoint();

        /**
        * @brief hash function
        * @return hash value
        */
        size_t hashFunction(
            int index    //!< [in] index-th feature
        );

    };
}}

#endif
