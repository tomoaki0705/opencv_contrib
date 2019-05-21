#ifndef __OPENCV_BDH_HPP__
#define __OPENCV_BDH_HPP__

#include <opencv2/core.hpp>

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
    CV_EXPORTS_W bool readBinary(const String &path, unsigned &dim, unsigned &num, featureElement** &data);

    typedef unsigned collision_t;//!< type of collision
    typedef char* address_t;	 //!< type of address

                                 /**
                                 * @brief a chain list
                                 */
    struct bin_t
    {
        collision_t collision;		 //!< collision
        address_t addressOfChainList;//!< head address of chain list
    };

    /**
    * @brief hash table
    */
    class HashTable
    {

    private:
        static const size_t collisionSize = sizeof(collision_t); //!< bytes of collision

        size_t entrySize;	 //!< bytes of entry
        size_t hashSize;	 //!< size of hash table
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
        HashTable(size_t entrySize, size_t hashSize)
            : entrySize(entrySize)
            , hashSize(hashSize)
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
            size_t hashSize	 //!< [in] hash size = 1<<bit;
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
            size_t HashValue,	 //!< [in] hash value of point data
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
            size_t HashValue,	//!< [in] hash value of point data
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
        );

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

    struct base_t
    {
        static int dim;		//!< dimension of data space (static member)

        int idx;			//!< index of base
        double mean;		//!< mean at base direction
        double variance;	//!< variance at base direction
        double* direction;	//!< direction of base [dim]

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

        int idx;				//!< index of base set
        int subDim;				//!< number of base
        double variance;		//!< sum of variance of base

        base_t* base;			//!< set of base_t[subDim]

        int k;					//!< number of centorid = 2^bit
        double bit;				//!< amount of infomation for this base set = log2(k)
        double error;			//!< error of quantization = sum of cellVariance
        double score;			//!< efficiency for incrementint the number of centroid
        double** centroid;		//!< centroid[k][subDim]
        double* cellVariance;	//!< variance in cell[k]

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

    class Subspace
    {

    public:
        static int dim;		 //!< dimension
        int subDim;			 //!< dimension of subspace
        int subHashSize;	 //!< hash size at subspace = 1<<bit
        double bit;			 //!< information volume
        double variance;	 //!< sum of variance
        double** base;		 //!< base direction[P][dim]
        size_t* hashKey;	 //!< hash value of bin corespond to centroid[subHashSize]
        double* cellVariance;//!< variance in cell[subHashSize]
        double** centroid;	 //!< centroid[subHashSize][subDim]

    public:

        /**
        * @brief default constructor
        */
        Subspace()
            : subDim(0)
            , subHashSize(0)
            , bit(0.0)
            , variance(0.0)
            , base(nullptr)
            , hashKey(nullptr)
            , cellVariance(nullptr)
            , centroid(nullptr)
        {}

        ~Subspace()
        {
            if (base != nullptr)
            {
                for (int p = 0; p < subDim; ++p)
                {
                    delete[] base[p];
                }
                delete[] base;
            }

            delete[] hashKey;

            delete[] cellVariance;

            if (centroid != nullptr)
            {
                for (int b = 0; b < subHashSize; ++b)
                {
                    delete[] centroid[b];
                }
                delete[] centroid;
            }
        }

        /**
        * @brief initialize all member variables
        */
        void clear();

        /**
        * @brief set training paramters
        */
        void setParameters(
            const baseset_t& baseSet
        );

        /**
        * @brief inner product
        */
        template<typename data_t>
        double innerProduct(
            double* base,
            const data_t* data
        ) const;

        /**
        * @brief project data into Principal Component space
        */
        template<typename data_t>
        void getPCAdata(
            const data_t* data,
            double* PCAdata) const;

        /**
        * @brief get sub hash value
        */
        template<typename data_t>
        size_t getSubHashValue(
            const data_t* data
        ) const;

        /**
        * @brief set node param
        */
        template<typename data_t>
        void setNodeParam(
            node_t* node,
            data_t* query
        );

        /**
        * @brief get distance to centroid
        */
        double getDistanceToCentroid(
            double* PCAquery,
            int centroidIndex
        )const;
    };

    class Index
    {
    public:
        typedef unsigned index_t;//!< type of index for point

    private:

        int dim;			//!< dimension of dataspace
        int M;				//!< number of subspace
        int P;				//!< dimension of subspace
        int bit;			//!< bits num of hash table
        double delta;		//!< increment step of search radius for C search
        int subHashSizeMax; //!< max of sub hash size
        size_t pointSize;	//!< number of data points
        size_t entrySize;	//!< size of entry = sum of size of index and data point
        size_t hashSize;	//!< hash size = 2^bit
        double variance;

        Subspace* subspace;	//!< classes handling parameters of subspace
        Subspace  lestspace;//!< classe handling parameters of subspace not which construct the hash table

        HashTable hashTable;	//!< hash table

    public:

        /**
        * @brief default constructor
        */
        Index()
            : dim(0)
            , M(0)
            , bit(0)
            , delta(0.0)
            , pointSize(0)
            , entrySize(0)
            , hashSize(0)
            , subspace(nullptr)
            , hashTable()
        {}

        Index(int dim, unsigned num, featureElement** data);

        ~Index()
        {
            delete[] subspace;
        }

        size_t get_nDdataPoints() const
        {
            return hashTable.get_nEntry();
        }

        double get_variance() const
        {
            return variance;
        }

        ////////////////parameterTuning///////////////

        /**
        * @brief parameterTuning parameters for hashing
        */
        void parameterTuning(
            int dim,				//!< [in] dimension of data space
            index_t num,			//!< [in] number of data points
            featureElement** const data,	//!< [in] sample points for training
            base_t* const base,		//!< [in] base for projectToPCspace
            int M,					//!< [in] number of subspace
            int P,					//!< [in] dimension of subspace
            int bit,				//!< [in] bits num of hash table
            double bit_step = 1.0,	//!< [in] training parameter. 0 < bit_step <= bit.
            double sampling_rate = 1.0		//!< [in] training parameter.  0 < rate <= 1.
        );

        /**
        * @brief parameterTuning parameters for hashing
        */
        void parameterTuning_ICCV2013(
            int dim,				//!< [in] dimension of data space
            index_t num,			//!< [in] number of data points
            featureElement** const data,	//!< [in] sample points for training
            base_t* const base,		//!< [in] base for projectToPCspace
            int P,					//!< [in] dimension of subspace
            int bit,				//!< [in] bits num of hash table
            double bit_step = 1.0,	//!< [in] training parameter. 0 < bit_step <= bit.
            double sampling_rate = 1.0		//!< [in] training parameter.  0 < rate <= 1.
        );

        ////////// store data points /////////////////

        /**
        * @brief store point set into hash table
        */
        void storePoint(
            index_t num,	//!< [in] number of data points. 
            featureElement**data	//!< [in] data point set. 
        );

        /**
        * @brief hash function
        * @return hash value
        */
        size_t hashFunction(
            featureElement* data	//!< [in] a point 
        );

        private:
        void setParameters(const baseset_t * const baseSet, const baseset_t & lestSet);

    };
}}

#endif
