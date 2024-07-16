!/bin/bash

version=cfic
PROJECT_DIR="$(dirname "${PWD}")" && readonly PROJECT_DIR
topk=3

echo $version
checkpoint="${PROJECT_DIR}/data/checkpoints"

# inference
for i in hotpotqa musique 2wikimqa multifieldqa_en qasper narrativeqa;  
do  
echo inference $i  ;
sh infer.sh $version $i $checkpoint $topk
done  

for i in llama2-7b-chat-4k vicuna-v1.5-7b-16k; 
# for i in llama2-7b-chat-4k;  
do  
echo pred with $i  ;
sh pred.sh $i $version 
done  

for i in llama2-7b-chat-4k vicuna-v1.5-7b-16k;  
# for i in llama2-7b-chat-4k;  
do  
echo evalualete $i  ;
sh eval.sh $i $version
