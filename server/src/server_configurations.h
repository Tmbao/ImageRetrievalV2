#ifndef server_configurations_h
#define server_configurations_h

#include <string>


namespace configs {
// Global params
const size_t N_THREADS = 8;
const size_t BATCH_SIZE = 128;

// Quantization params
const std::string CODEBOOK_FILE = "/Users/tmbao/Sources/v-commerce/fixture/Clustering_l2_1000000_13516675_128_50it.hdf5";
const std::string CODEBOOK_NAME = "clusters";
const std::string INDEX_FILE = "/Users/tmbao/Sources/v-commerce/fixture/index.hdf5";

// Database params
const size_t N_WORDS = 1000000;
const std::string IMAGE_FOLDER = "/Users/tmbao/Sources/v-commerce/fixture/images";
const std::string CACHE_FOLDER = "/Users/tmbao/Sources/v-commerce/fixture/cache";
}

#endif /* server_configurations_h */
