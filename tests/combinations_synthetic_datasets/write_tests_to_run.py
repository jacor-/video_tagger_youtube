
### BE CAREFUL: FOR LOGGING PURPOSES
# We will log each nuber of iterations 'I'. This will be dynamic across tests depending on the frames per video.
# The reason is that a video with more frames will see more images per each iteration so it would not be fair to compare the convergence time. This way we will solve this... I guess
# Simply take into account this when you do the fucking plots! If you want epochs you will need to take some samples only

#### Example execution commands
### THEANO_FLAGS="dnn.enabled=True" PYTHONPATH=. python tests/combinations_synthetic_datasets/runtest.py  --name=simple_experiment --basenet=resnet_full_wide --multiclass=False --dataset=cifar-10 --log_epochs=1 --nepochs=50000 --aggregation=tanh_aggregation --base_train_examples=100 --base_test_examples=100 --frames=3 --batch=50 --snapshot_epochs=100
### THEANO_FLAGS="dnn.enabled=True" PYTHONPATH=. python tests/combinations_synthetic_datasets/runtest.py  --name=simple_experiment --basenet=resnet_full --multiclass=False --dataset=cifar-10 --log_epochs=1 --nepochs=50000 --aggregation=tanh_aggregation --base_train_examples=100 --base_test_examples=100 --frames=3 --batch=5 --snapshot_epochs=100
### THEANO_FLAGS="dnn.enabled=True" PYTHONPATH=. python tests/combinations_synthetic_datasets/runtest.py  --name=mnist_experiment --basenet=simple_cnn --multiclass=True --dataset=mnist --log_epochs=10 --nepochs=50000 --aggregation=max_aggregation --base_train_examples=40000 --base_test_examples=2500 --frames=3 --batch=25 --snapshot_epochs=100

#### Let's build the battery of tests

import os
import time

## MNIST TESTS
multiclass = False
test_aggregators = ['max_aggregation', 'poissonbernoulli_aggregation']
test_frames = [2,5,8]
experiment_name = "mnist_test_%s"
train_examples = 40000
batch = 25
base_mnist_test = 'THEANO_FLAGS="dnn.enabled=True" PYTHONPATH=. python tests/combinations_synthetic_datasets/runtest.py  --name={experiment_name} --basenet=simple_cnn --multiclass={multiclass} --dataset=mnist --log_epochs={log_iters} --nepochs=50 --aggregation={aggregator} --base_train_examples={train_examples} --base_test_examples=1200 --frames={n_frames} --batch={batch} --snapshot_epochs=74'

# 1 - MNIST BASELINE
log_per_iter = int(round(float(train_examples*10) / batch) / 1)
test_command = base_mnist_test.format(n_frames = 1, aggregator = 'max_aggregation', experiment_name = experiment_name % 'baseline', multiclass = str(multiclass), batch = batch, train_examples = train_examples, log_iters = log_per_iter)
print(test_command)

# 2 - MNIST TESTS
for n_frames in test_frames:
    for aggregation_type in test_aggregators:
        log_per_iter = int(round(float(train_examples*10) / batch) / n_frames)
        test_command = base_mnist_test.format(n_frames = n_frames, aggregator = aggregation_type, experiment_name = experiment_name % (str(n_frames) + "_" + aggregation_type), multiclass = str(multiclass), batch = batch, train_examples = train_examples, log_iters = log_per_iter)
        print(test_command)

## CIFAR-10 TESTS

multiclass = False
test_aggregators = ['max_aggregation','poissonbernoulli_aggregation']
test_frames = [2,5,8]
experiment_name = "cifar_10_test_%s"
train_examples = 100000
batch = 25
base_mnist_test = 'THEANO_FLAGS="dnn.enabled=True" PYTHONPATH=. python tests/combinations_synthetic_datasets/runtest.py  --name={experiment_name} --basenet=simple_cnn --multiclass={multiclass} --dataset=cifar-10 --log_epochs={log_iters} --nepochs=250 --aggregation={aggregator} --base_train_examples={train_examples} --base_test_examples=1000 --frames={n_frames} --batch={batch} --snapshot_epochs=99'

# 1 - CIFAR-10 BASELINE
log_per_iter = int(round(float(train_examples*10) / batch) / 1)
test_command = base_mnist_test.format(n_frames = 1, aggregator = 'max_aggregation', experiment_name = experiment_name % 'baseline', multiclass = str(multiclass), batch = batch, train_examples = train_examples, log_iters = log_per_iter)
print(test_command)

# 2 - CIFAR-10 TESTS
for n_frames in test_frames:
    for aggregation_type in test_aggregators:
        log_per_iter = int(round(float(train_examples*10) / batch) / n_frames)
        test_command = base_mnist_test.format(n_frames = n_frames, aggregator = aggregation_type, experiment_name = experiment_name % (str(n_frames) + "_" + aggregation_type), multiclass = str(multiclass), batch = batch, train_examples = train_examples, log_iters = log_per_iter)
        print(test_command)

## CIFAR-100 TESTS

multiclass = False
test_aggregators = ['max_aggregation','poissonbernoulli_aggregation']
test_frames = [2,5,8]
experiment_name = "cifar_100_test_%s"

base_mnist_test = 'THEANO_FLAGS="dnn.enabled=True" PYTHONPATH=. python tests/combinations_synthetic_datasets/runtest.py  --name={experiment_name} --basenet=simple_cnn --multiclass={multiclass} --dataset=cifar-100 --log_epochs={log_iters} --nepochs=500 --aggregation={aggregator} --base_train_examples={train_examples} --base_test_examples=1000 --frames={n_frames} --batch={batch} --snapshot_epochs=99'

# 1 - CIFAR-10 BASELINE
log_per_iter = int(round(float(train_examples*10) / batch) / 1)
test_command = base_mnist_test.format(n_frames = 1, aggregator = 'max_aggregation', experiment_name = experiment_name % 'baseline', multiclass = str(multiclass), batch = batch, train_examples = train_examples, log_iters = log_per_iter)
print(test_command)

# 2 - CIFAR-10 TESTS
for n_frames in test_frames:
    for aggregation_type in test_aggregators:
        log_per_iter = int(round(float(train_examples*10) / batch) / n_frames)
        test_command = base_mnist_test.format(n_frames = n_frames, aggregator = aggregation_type, experiment_name = experiment_name % (str(n_frames) + "_" + aggregation_type), multiclass = str(multiclass), batch = batch, train_examples = train_examples, log_iters = log_per_iter)
        print(test_command)
