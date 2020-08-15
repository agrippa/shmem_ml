#ifndef _SHMEM_ML_UTILS_HPP
#define _SHMEM_ML_UTILS_HPP

#include <iostream>

#define CHECK_ARROW(call) { \
    arrow::Status err = (call); \
    if (!err.ok()) { \
        std::cerr << "Arrow error @ " << std::string(__FILE__) << ":" << \
                __LINE__ << " - " << err.ToString() << std::endl; \
        abort(); \
    } \
}

#endif
