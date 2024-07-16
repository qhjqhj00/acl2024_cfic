PWD="$(pwd)" && readonly PWD
PROJECT_DIR="$(dirname "${PWD}")" && readonly PROJECT_DIR

python $PROJECT_DIR/src/eval.py --model $1
