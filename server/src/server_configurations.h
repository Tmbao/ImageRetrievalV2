#ifndef server_configurations_h
#define server_configurations_h

#include <string>


namespace configs {
// Global params
const size_t N_THREADS = 6;
const size_t BATCH_SIZE = 128;

// Quantization params
const std::string CODEBOOK_FILE = "/Volumes/ExternalDisk/fixture/Clustering_l2_1000000_13516675_128_50it.hdf5";
const std::string CODEBOOK_NAME = "clusters";
const std::string INDEX_FILE = "/Volumes/ExternalDisk/fixture/index.hdf5";

// Database params
const size_t N_WORDS = 1000000;
const std::string IMAGE_FOLDER = "/Volumes/ExternalDisk/fixture/images";
const std::string CACHE_FOLDER = "/Volumes/ExternalDisk/fixture/cache";
}

#endif /* server_configurations_h */
