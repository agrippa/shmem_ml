import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import horovod.tensorflow as hvd

start_setup = time.time()

hvd.init()

batch_size = 128
nthreads = 1
tf.config.threading.set_inter_op_parallelism_threads(nthreads)
tf.config.threading.set_intra_op_parallelism_threads(nthreads)

# Distributed, collective allocation of a 10-element array of float64
# nsamples = 5000000
nsamples = 5000000
nfeatures = 5

samples_per_rank = int((nsamples + hvd.size() - 1) / hvd.size())
nsamples = samples_per_rank * hvd.size()
my_start_sample = hvd.rank() * samples_per_rank
my_end_sample = (hvd.rank() + 1) * samples_per_rank
n_my_samples = my_end_sample - my_start_sample

vec = np.zeros(n_my_samples)
mat = np.zeros((n_my_samples, nfeatures))

for i in range(my_start_sample, my_end_sample):
    vec[i - my_start_sample] = 10.0 * i / nsamples
    for j in range(nfeatures):
        mat[i - my_start_sample, j] = 10.0 * i / nsamples if j == 0 else 0.


max_print_lines = 20
if hvd.rank() == 0:
    print('Running in horovod job of ' + str(hvd.size()) + ' w/ ' +
            str(nsamples) + ' samples')
    print('Labels:')
    for i in range(max_print_lines if len(vec) > max_print_lines else len(vec)):
        print('  ' + str(vec[i]))
    if len(vec) > max_print_lines:
        print('  ...')
    for i in range(len(vec) - max_print_lines if len(vec) - max_print_lines >= 0 else 0, len(vec)):
        print('  ' + str(vec[i]))
    print('')
    print('Features:')
    for i in range(max_print_lines if mat.shape[0] > max_print_lines else mat.shape[0]):
        sys.stdout.write(' ')
        for j in range(3 if mat.shape[1] > 3 else mat.shape[1]):
            # print('  (' + str(i) + ', ' + str(j) + ') ' + str(mat.get(i, j)))
            sys.stdout.write(' ' + str(mat[i, j]))
        if mat.shape[1] > 3:
            sys.stdout.write(' ...')
        print('')
    if mat.shape[0] > max_print_lines:
        print('  ...')
    print('')


clf = keras.Sequential()
clf.add(tf.keras.Input(shape=(nfeatures,)))
# clf.add(tf.keras.layers.Dense(2, activation='relu',
#     kernel_initializer=tf.keras.initializers.GlorotNormal(seed=40),
#     bias_initializer=tf.keras.initializers.GlorotNormal(seed=400)))
clf.add(tf.keras.layers.Dense(1, activation='relu',
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=42),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=43),
    dtype='float64'))
opt = keras.optimizers.SGD(learning_rate=0.005)
# opt = tf.optimizers.Adam(0.001 * hvd.size())

# loss = tf.losses.SparseCategoricalCrossentropy()
loss = tf.losses.MeanSquaredError()
clf.compile(optimizer=opt, loss=loss)

@tf.function
def training_step(images, labels, first_batch):
    with tf.GradientTape() as tape:
        probs = clf(mat, training=True)
        loss_value = loss(vec, probs)

    # Horovod: add Horovod Distributed GradientTape.
    tape = hvd.DistributedGradientTape(tape)

    grads = tape.gradient(loss_value, clf.trainable_variables)
    opt.apply_gradients(zip(grads, clf.trainable_variables))

    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    #
    # Note: broadcast should be done after the first gradient step to ensure optimizer
    # initialization.
    if first_batch:
        hvd.broadcast_variables(clf.variables, root_rank=0)
        hvd.broadcast_variables(opt.variables(), root_rank=0)

    return loss_value

start_train = time.time()
# Horovod: adjust number of steps based on number of GPUs.

# niters = 100
niters = 100
for it in range(niters):
    start_iter = time.time()
    for i in range(0, n_my_samples, batch_size):
        loss_value = training_step(mat[i:i+batch_size, :], vec[i:i+batch_size], it == 0 and i == 0)
    # loss_value = training_step(mat[:, :], vec[:], it == 0)
    elapsed_iter = time.time() - start_iter
    if hvd.rank() == 0:
        print('Epoch #%d\trank %d\t%f s\tLoss: %.6f' % (it, hvd.rank(), elapsed_iter, loss_value))

end_train = time.time()
pred = clf.predict(mat)
end_predict = time.time()

if hvd.rank() == 0:
    print('PE=' + str(hvd.rank()) + ' sees predictions, # iters=' + str(niters))
    for i in range(max_print_lines if len(pred) > max_print_lines else len(pred)):
        print('  ' + str(pred[i, 0]))
    if len(pred) > max_print_lines:
        print('  ...')
    for i in range(len(pred) - max_print_lines if len(pred) - max_print_lines >= 0 else 0, len(pred)):
        print('  ' + str(pred[i, 0]))
    print('')
    print(clf._collected_trainable_weights)

    print('Distributed set up took ' + str(start_train - start_setup) + ' s')
    print('Distributed training took ' + str(end_train-start_train) + ' s, ' + str(niters) + ' iters')
    print('Distributed inference took ' + str(end_predict-end_train) + ' s')
