import sys
import time
import numpy as np
import pyarrow
from PyShmemML import shmem_ml_init, shmem_ml_finalize, PyShmemML1DD, PyShmemML2DD, rand, pe, npes, Sequential
import tensorflow
from tensorflow import keras

# Set up runtime
shmem_ml_init()

# Distributed, collective allocation of a 10-element array of float64
# nsamples = 5000000
nsamples = 500000
nfeatures = 5
vec = PyShmemML1DD(nsamples)
mat = PyShmemML2DD(nsamples, nfeatures)

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

vec = vec.apply(lambda i, x, vec: 10.0 * i / vec.N())
mat = mat.apply(lambda i, j, x, mat: 10.0 * i / mat.M() if j == 0 else 0.)

max_print_lines = 20
if pe() == 0:
    print('Labels:')
    for i in range(max_print_lines if vec.N() > max_print_lines else vec.N()):
        print('  ' + str(vec.get(i)))
    if vec.N() > max_print_lines:
        print('  ...')
    print('')
    print('Features:')
    for i in range(max_print_lines if mat.M() > max_print_lines else mat.M()):
        sys.stdout.write(' ')
        for j in range(3 if mat.N() > 3 else mat.N()):
            # print('  (' + str(i) + ', ' + str(j) + ') ' + str(mat.get(i, j)))
            sys.stdout.write(' ' + str(mat.get(i, j)))
        if mat.N() > 3:
            sys.stdout.write(' ...')
        print('')
    if mat.M() > max_print_lines:
        print('  ...')
    print('')

np.random.seed(2)
tensorflow.random.set_seed(33)
local_niters = 30
dist_niters = local_niters * 10
clf = Sequential()
clf.add(tensorflow.keras.Input(shape=(nfeatures,)))
# clf.add(tensorflow.keras.layers.Dense(2, activation='relu',
#     kernel_initializer=tensorflow.keras.initializers.GlorotNormal(seed=40),
#     bias_initializer=tensorflow.keras.initializers.GlorotNormal(seed=400)))
clf.add(tensorflow.keras.layers.Dense(1, activation='relu',
    kernel_initializer=tensorflow.keras.initializers.GlorotNormal(seed=42),
    bias_initializer=tensorflow.keras.initializers.GlorotNormal(seed=43)))
opt = keras.optimizers.SGD(learning_rate=0.01)
clf.compile(optimizer=opt, loss='mse')
vec.sync()

start_dist_fit = time.time()
clf.fit(mat, vec, epochs=dist_niters)
start_dist_pred = time.time()
pred = clf.predict(mat)
end_dist_pred = time.time()
pred.sync()

if pe() == 0:
    print('PE=' + str(pe()) + ' sees predictions with M=' + str(pred.M()) + ', # iters=' + str(dist_niters))
    for i in range(max_print_lines if pred.M() > max_print_lines else pred.M()):
        print('  ' + str(pred.get(i, 0)))
    if pred.M() > max_print_lines:
        print('  ...')
    for i in range(pred.M() - max_print_lines if pred.M() - max_print_lines >= 0 else 0, pred.M()):
        print('  ' + str(pred.get(i, 0)))
    print('')
    print(clf.model._collected_trainable_weights)
    # print(clf.model.__dict__)
    # print('')

    gathered_lbls = vec.gather()
    gathered_features = mat.gather()

    np.random.seed(2)
    tensorflow.random.set_seed(33)
    keras_model = keras.Sequential()
    keras_model.add(tensorflow.keras.Input(shape=(nfeatures,)))
    # keras_model.add(tensorflow.keras.layers.Dense(2, activation='relu',
    #     kernel_initializer=tensorflow.keras.initializers.GlorotNormal(seed=40),
    #     bias_initializer=tensorflow.keras.initializers.GlorotNormal(seed=400)))
    keras_model.add(tensorflow.keras.layers.Dense(1, activation='relu',
        kernel_initializer=tensorflow.keras.initializers.GlorotNormal(seed=42),
        bias_initializer=tensorflow.keras.initializers.GlorotNormal(seed=43)))
    opt = keras.optimizers.SGD(learning_rate=0.01)
    keras_model.compile(optimizer=opt, loss='mse')

    start_local_fit = time.time()
    keras_model.fit(gathered_features, gathered_lbls, epochs=local_niters, verbose=0,
            shuffle=False, batch_size=32)
    start_local_pred = time.time()
    keras_pred = keras_model.predict(gathered_features)
    end_local_pred = time.time()

    err = keras_model.evaluate(gathered_features, gathered_lbls)

    print('Local predictions with keras and # iters=' + str(local_niters) + ':')
    for i in range(max_print_lines if pred.M() > max_print_lines else pred.M()):
        print('  ' + str(keras_pred[i, 0]))
    if pred.M() > max_print_lines:
        print('  ...')
    for i in range(pred.M() - max_print_lines if pred.M() - max_print_lines >= 0 else 0, pred.M()):
        print('  ' + str(keras_pred[i, 0]))
    #print(err)
    print('')
    print(keras_model._collected_trainable_weights)
    # print(keras_model.__dict__)
    # print('')
    print('PE ' + str(pe()) + '. Distributed training took ' + str(start_dist_pred - start_dist_fit) + ' s')
    print('PE ' + str(pe()) + '. Local training took ' + str(start_local_pred - start_local_fit) + ' s')
    print('PE ' + str(pe()) + '. Distributed inference took ' + str(end_dist_pred - start_dist_pred) + ' s')
    print('PE ' + str(pe()) + '. Local inference took ' + str(end_local_pred - start_local_pred) + ' s')
    print('')

#pred.sync();
vec.sync();

# shmem_ml_finalize()
