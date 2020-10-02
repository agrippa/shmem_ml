import os
import sys
import numpy
import pyarrow
import socket
from setuptools import setup, Extension

from Cython.Build import cythonize

shmem_libs = ['sma']
host = socket.gethostname()
if host.endswith('frontera.tacc.utexas.edu'):
    shmem_libs = ['shmem', 'shmemc-ucx', 'pmix', 'shmemu', 'shmemt', 'shcoll']

extensions = [
        Extension("PyShmemML", ['./src/shmem_ml/PyShmemML.pyx'],
            include_dirs=[numpy.get_include(),
                pyarrow.get_include(),
                'src/shmem_ml/',
                'src/shmem_ml/crc',
                os.path.join(os.getenv('OPENSHMEM_HOME'), 'include')
                ],
            libraries= shmem_libs + ['arrow', 'arrow_python', 'shmem_ml'],
            library_dirs=[os.path.join(os.getenv('OPENSHMEM_HOME'), 'lib64'),
                          os.path.join(os.getenv('OPENSHMEM_HOME'), 'lib'),
                          os.path.join(os.getenv('OPENPMIX_HOME'), 'lib'),
                          os.path.join(os.getenv('ARROW_HOME'), 'lib'),
                          './bin'],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])
        ]

setup(name='shmem_ml',
      ext_modules=cythonize(extensions))
