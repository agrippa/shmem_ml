import sys
import time
import numpy as np
import pyarrow
from PyShmemML import shmem_ml_init, shmem_ml_finalize, PyShmemML1DD, PyShmemML2DD, rand, pe, npes, SGDRegressor
import sklearn.linear_model

# Set up runtime
# shmem_ml_init()

# Distributed, collective allocation of a 10-element array of float64
nsamples = 500000
vec = PyShmemML1DD(nsamples)
mat = PyShmemML2DD(nsamples, 5)

# Clear all values in vec to 0.0
vec.clear(0)

# Global barrier
vec.sync()

# Every rank uses RMA to grab the 0-th element of vec
value_at_zero = vec.get(0)
assert value_at_zero == 0.
vec.sync()

if pe() == 0:
    print('PE=' + str(pe()) + ' N=' + str(vec.N()) + ' local slice start=' +
            str(vec.local_slice_start()) + ' local slice end=' +
            str(vec.local_slice_end()))

# Initialize vec with random values (in place)
vec = rand(vec)
mat = rand(mat)

vec = vec.apply(lambda i, x: i / vec.N())
mat = mat.apply(lambda i, j, x: i / mat.M())

if pe() == 0:
    print('Labels:')
    for i in range(10 if vec.N() > 10 else vec.N()):
        print('  ' + str(vec.get(i)))
    if vec.N() > 10:
        print('  ...')
    print('')
    print('Features:')
    for i in range(10 if mat.M() > 10 else mat.M()):
        sys.stdout.write(' ')
        for j in range(3 if mat.N() > 3 else mat.N()):
            # print('  (' + str(i) + ', ' + str(j) + ') ' + str(mat.get(i, j)))
            sys.stdout.write(' ' + str(mat.get(i, j)))
        if mat.N() > 3:
            sys.stdout.write(' ...')
        print('')
    if mat.M() > 10:
        print('  ...')
    print('')

niters = 10
clf = SGDRegressor(max_iter=niters)
vec.sync()

start_dist_fit = time.time()
clf.fit(mat, vec)
elapsed_dist_fit = time.time() - start_dist_fit
pred = clf.predict(mat)

if pe() == 0:
    print('PE=' + str(pe()) + ' sees predictions with N=' + str(pred.N()) + ', # iters=' + str(niters))
    for i in range(10 if pred.N() > 10 else pred.N()):
        print('  ' + str(pred.get(i)))
    if pred.N() > 10:
        print('  ...')
    print('')

    gathered_lbls = vec.gather()
    gathered_features = mat.gather()
    sk_model = sklearn.linear_model.SGDRegressor(max_iter=niters, tol=None)
    start_local_fit = time.time()
    sk_model.fit(gathered_features, gathered_lbls)
    elapsed_local_fit = time.time() - start_local_fit
    sk_pred = sk_model.predict(gathered_features)
    print('Local predictions with sklearn and # iters=' + str(niters) + ':')
    for i in range(10 if pred.N() > 10 else pred.N()):
        print('  ' + str(sk_pred[i]))
    if pred.N() > 10:
        print('  ...')
    print('PE ' + str(pe()) + '. Distributed training took ' + str(elapsed_dist_fit) + ' s')
    print('PE ' + str(pe()) + '. Local training took ' + str(elapsed_local_fit) + ' s')
    print(sk_model.__dict__)

pred.sync();


# Shut down runtime
#shmem_ml_finalize()
