# shmem_ml

## Environment:

On Cori:

    module unload PrgEnv-intel
    module load PrgEnv-gnu

    module load cdt/20.06
    module load cray-openshmemx/9.1.0
    module load python/3.7-anaconda-2019.10

    export OPENSHMEM_HOME=$CRAY_OPENSHMEMX_DIR

    export LD_LIBRARY_PATH=$OPENSHMEM_HOME/lib64:$LD_LIBRARY_PATH

    export ARROW_HOME=$HOME/.conda/envs/pyarrow-dev/
    export LD_LIBRARY_PATH=/lib64:$LD_LIBRARY_PATH:$ARROW_HOME/lib

## Dependencies:

- Apache Arrow C++ and Python APIs
  - https://github.com/apache/arrow

## Building Arrow on Cori

    module unload PrgEnv-intel
    module load PrgEnv-gnu

    module load cdt/20.06
    module load python/3.7-anaconda-2019.10

    mkdir $HOME/arrow
    pushd $HOME/arrow
    git clone https://github.com/apache/arrow.git arrow-src

    pushd arrow-src
    git submodule init
    git submodule update
    export PARQUET_TEST_DATA="${PWD}/cpp/submodules/parquet-testing/data"
    export ARROW_TEST_DATA="${PWD}/testing/data"

    conda create -y -n pyarrow-dev -c conda-forge \
            --file arrow-src/ci/conda_env_unix.yml \
            --file arrow-src/ci/conda_env_cpp.yml \
            --file arrow-src/ci/conda_env_python.yml \
            --file arrow-src/ci/conda_env_gandiva.yml \
            python=3.7 \
            pandas

    source activate pyarrow-dev
    export ARROW_HOME=$CONDA_PREFIX

    mkdir cpp/build
    pushd cpp/build
    CC=$(which gcc) CXX=$(which g++) cmake -DCMAKE_INSTALL_PREFIX=$ARROW_HOME \
                                 -DCMAKE_INSTALL_LIBDIR=lib \
                                 -DARROW_WITH_BZ2=ON \
                                 -DARROW_WITH_ZLIB=ON \
                                 -DARROW_WITH_ZSTD=ON \
                                 -DARROW_WITH_LZ4=ON \
                                 -DARROW_WITH_SNAPPY=ON \
                                 -DARROW_WITH_BROTLI=ON \
                                 -DARROW_PARQUET=ON \
                                 -DARROW_PYTHON=ON \
                                 -DARROW_BUILD_TESTS=ON \
                                 ..
                                 make -j4
                                 make install
    popd

    pushd python
    export PYARROW_WITH_PARQUET=1
    CC=$(which gcc) CXX=$(which g++) python setup.py build_ext --inplace
    popd

