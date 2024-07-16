PWD="$(pwd)" && readonly PWD
PROJECT_DIR="$(dirname "${PWD}")" && readonly PROJECT_DIR

version=$1
dataset=$2
cp_path=$3
topk=$4

refined_path=$PROJECT_DIR/data/refined/$version

if [ ! -d $refined_path ]; then
  mkdir $refined_path
  echo "$refined_path is not exist, make directory"
fi

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python $PROJECT_DIR/src/infer.py \
    --eval_file $PROJECT_DIR/data/chunks/refine_0119/$dataset.chunk.json \
    --checkpoint_path $cp_path \
    --save_path $refined_path/$dataset.refined.jsonl \
    --raw_file $PROJECT_DIR/data/longbench/$dataset.jsonl \
    --batch_size  1 \
    --flash_attn True \
    --max_length 32768 \
    --n_sample $topk \
    --max_new_token 32 \
    --min_new_token 8  \
    --post_process False \
    --sample_sents True

