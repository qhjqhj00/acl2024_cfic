model_name=$1
version=$2
PWD="$(pwd)" && readonly PWD
PROJECT_DIR="$(dirname "${PWD}")" && readonly PROJECT_DIR

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python $PROJECT_DIR/src/infer.py --model $model_name --v $version
