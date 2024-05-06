#!/bin/bash

DIR=`pwd`

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

#mkdir -p $DIR/logs  # 创建logs目录
 


#DATASET_1="<PATH TO THE FIRST DATASET>"
#DATASET_2="<PATH TO THE SECOND DATASET>"
#DATASET_3="<PATH TO THE THIRD DATASET>"
#DATASET="0.2 ${DATASET_1} 0.3 ${DATASET_2} 0.5 ${DATASET_3}"
#DATASET=`python get_data_path.py`

DATASET="/data/train_data/"

BASE_DATA_PATH=.
#DATASET="1 /raid/nlp/processed_data/zhiyuan/part-2021278643_content_document 1 /raid/nlp/processed_data/THUCNews/股票/647234_text_document"
VOCAB_PATH=${BASE_DATA_PATH}/ch_tokenizer_data/vocab.txt
SPECIAL_TOKEN_PATH=${BASE_DATA_PATH}/ch_tokenizer_data/special_tokens.yaml

script_path=$(realpath $0)
# echo $script_path

script_dir=$(dirname $script_path)

CONFIG_JSON="$script_dir/ds_config.json"

USE_DEEPSPEED=1
ZERO_STAGE=0


# Debug
#TP=4
#PP=4
#LAYERS=8
#HIDDEN=512
#SEQ=1024
#GLOBAL_BATCH=128
#WORKER_STR="-i worker-0"


# 1.7B
TP=1
PP=8
HIDDEN=6144
LAYERS=64
NUM_ATTENTION_HEADS=32
SEQ=8192
GLOBAL_BATCH=1024
WORKER_STR=""

MICRO_BATCH=1

#ls
# export NCCL_SOCKET_IFNAME=enp226s0,ib,eth
#export NCCL_SOCKET_NTHREADS=16
#export NCCL_IB_DISABLE=0
#export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1

CHECKPOINT_PATH=/data/checkpoints/pipegpt_test1


while [ $# -gt  0 ]
do
key="$1"
case $key in
    --no-deepspeed)
    USE_DEEPSPEED=0;
    shift
    ;;
    -z|--zero-stage)
    ZERO_STAGE=$2;
    shift
    ;;
    *)
    echo "Unknown argument(s)"
    usage
    exit 1
    shift
    ;;
esac
done


options=" \
	--tensor-model-parallel-size $TP \
	--pipeline-model-parallel-size $PP \
    --num-layers $LAYERS \
    --hidden-size $HIDDEN \
    --num-attention-heads $NUM_ATTENTION_HEADS \
    --seq-length $SEQ \
    --loss-scale 12 \
    --max-position-embeddings $SEQ \
    --micro-batch-size $MICRO_BATCH \
    --global-batch-size $GLOBAL_BATCH \
    --train-iters 200000 \
    --lr 2e-4 \
	  --min-lr 2.0e-5 \
    --lr-warmup-iters 1000 \
    --lr-decay-style cosine \
    --lr-decay-iters 10000 \
    --log-interval 1 \
    --eval-iters 40 \
    --eval-interval 1000000 \
    --data-path ${DATASET} \
    --vocab-file ${VOCAB_PATH} \
    --special-token-file ${SPECIAL_TOKEN_PATH} \
    --save ${CHECKPOINT_PATH} \
    --load ${CHECKPOINT_PATH} \
	  --save-interval 100 \
    --split 800,100,100 \
    --clip-grad 1.0 \
    --optimizer adam \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.006 \
    --tokenizer-type BaichuanTokenizer \
    --tokenizer-name-or-path decapoda-research/llama-7b-hf \
    --tensorboard-dir $CHECKPOINT_PATH/tensorboard_dir \
    --init-weight True \
    --fp16 \
    --use-flash-attn \
	  --checkpoint-activations
        "
#echo $USE_DEEPSPEED

if [ ${USE_DEEPSPEED} -eq 1 ]; then
	echo "Using DeepSpeed"
	options="${options} \
		--deepspeed \
		--deepspeed_config=${CONFIG_JSON} \
		--zero-stage=${ZERO_STAGE} \
		--deepspeed-activation-checkpointing \
	"
fi
#echo $options
#exit

cat <<EOT > $CONFIG_JSON
{
  "train_batch_size" : $GLOBAL_BATCH,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": 1,

  "zero_optimization": {
    "stage": 1,
    "contiguous_gradients":true,
    "overlap_comm":true,
    "reduce_scatter":true,
    "reduce_bucket_size":5e7,
    "allgather_bucket_size":5e7,
    "cpu_offload": true
  },
  "gradient_clipping":1.0,
  "prescale_gradients":false,

  "fp16": {
    "enabled": true,
    "loss_scale": 4096,
    "loss_scale_window": 50000,
    "hysteresis": 2,
    "min_loss_scale": 1024,
    "initial_scale_power": 14
  },


  "wall_clock_breakdown" : true
}
EOT

#run_cmd="deepspeed -i worker-0:0,1,2,3 ${DIR}/pretrain_gpt.py $@ ${options}"
#run_cmd="deepspeed -i worker-0 ${DIR}/pretrain_gpt.py $@ ${options}"
#run_cmd="CUDA_VISIBLE_DEVICES=1  deepspeed   ${DIR}/pretrain_gpt.py $@ ${options}"
#run_cmd="CUDA_VISIBLE_DEVICES=0 deepspeed ${DIR}/pretrain_gpt.py $@ ${options}"

run_cmd="nohup deepspeed   ${DIR}/pretrain_gpt.py $@ ${options} > log_01_03.txt 2>&1 & tail -f log_01_03.txt"


echo ${run_cmd}
eval ${run_cmd}
#eval nohup ${run_cmd} & tail -f nohup.out

set +x

