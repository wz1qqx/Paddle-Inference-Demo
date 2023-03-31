#!/bin/bash
set +x
set -e

work_path=$(dirname $(readlink -f $0))
model_dir=${work_path}/../../../Models
model_name=inference_model/model.pdmodel
param_name=inference_model/model.pdiparams

# 1. compile
if [ ! -d ${work_path}/build ]; then
    echo "compile test.cc!"
    bash ${work_path}/compile.sh
fi
# 2. download model
# if [ -d resnet50 ]; then
#     wget https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz
#     tar xzf resnet50.tgz
# fi

# 3. run
./build/test --model_file ${model_dir}/hrnet/${model_name} --param_file ${model_dir}/hrnet/${param_name} --input_shapes "1,3,512,960"
# ./build/test --model_file ${model_dir}/paddle_appearance/model.pdmodel --param_file ${model_dir}/paddle_appearance/model.pdiparams --input_shapes "1,64,64,3"
