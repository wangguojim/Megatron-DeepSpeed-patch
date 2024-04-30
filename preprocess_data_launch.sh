#!/bin/bash

start_time=`date +%s`              #定义脚本运行的开始时间
[ -e /tmp/fd1 ] || mkfifo /tmp/fd1 #创建有名管道
exec 3<>/tmp/fd1                   #创建文件描述符，以可读（<）可写（>）的方式关联管道文件，这时候文件描述符3就有了有名管道文件的所有特性
rm -rf /tmp/fd1                    #关联后的文件描述符拥有管道文件的所有特性,所以这时候管道文件可以删除，我们留下文件描述符来用就可以了


INPUT_DIR=/data/nvme3/pretrain_data/Megatron-data/json_files
OUTPUT_DIR=/data/nvme3/pretrain_data/Megatron-data/bin_files
rm -rf $OUTPUT_DIR/*.bin
rm -rf $OUTPUT_DIR/*.idx


#exit
for INPUT_FILE in `ls $INPUT_DIR`
do
  read -u3  #代表从管道中读取一个令牌
  {
  INPUT_FILE=${INPUT_FILE%.*}

    python tools/preprocess_data.py \
         --input $INPUT_DIR/$INPUT_FILE.json \
         --output-prefix $OUTPUT_DIR/$INPUT_FILE \
         --tokenizer_path /data/nvme3/checkpoints/Qwen1.5-7B \
         --tokenizer-type QwenTokenizer\
         --json-keys 'text' \
         --dataset-impl mmap \
         --workers 32
    echo >&3

}&
done

wait

stop_time=`date +%s`  #定义脚本运行的结束时间

echo "TIME:`expr $stop_time - $start_time`"
exec 3<&-                       #关闭文件描述符的读
exec 3>&-                       #关闭文件描述符的写
