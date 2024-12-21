#1 /bin/bash

if [[ -z $MNIST_ML_ROOT ]]; then
    echo "The variable MNIST_ML_ROOT is not defined"
    exit 1
fi

model_name=$(echo "$@" | tr A-Z a-z)

mkdir -p $MNIST_ML_ROOT/$model_name/include $MNIST_ML_ROOT/$model_name/src
touch $MNIST_ML_ROOT/$model_name/Makefile
touch $MNIST_ML_ROOT/$model_name/include/"$model_name.hpp"
touch $MNIST_ML_ROOT/$model_name/src/"$model_name.cc"