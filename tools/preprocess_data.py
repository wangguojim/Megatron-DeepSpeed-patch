# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Processing data for pretraining."""

import argparse
import json
import multiprocessing
import os
import sys
import time
import yaml
import time
import torch

sys.path.append('/data/nvme3/Megatron-DeepSpeed-patch/')
sys.path.append('/data/nvme3/Megatron-DeepSpeed/')


from megatron_patch.tokenizer.tokenizer  import build_tokenizer
from megatron.data import indexed_dataset



class IdentitySplitter(object):
    def tokenize(self, *text):
        return text



class Encoder:
    def __init__(self, args):
        self.args = args
        self.tokenizer=build_tokenizer(args.tokenizer_path,
                                       args.tokenizer_type)
    def initializer(self):
        #             self.tokenizer = BertTokenizer.from_pretrained(self.args.vocab_file)
        pass

    def encode(self, json_line):
        data = json.loads(json_line.strip('\n'), strict=False)
        ids = {}
        key='text'
        text=data[key]
        doc_ids = []

        sentence_ids = self.tokenizer.encode(text)
        if len(sentence_ids) > 0:
            doc_ids.append(sentence_ids[1:-1])  # 编码后的句子存入doc_ids
        if len(doc_ids) > 0 and self.args.append_eod:
            if self.args.tokenizer_type == 'BloomTokenizer':
                doc_ids[-1].append(self.tokenizer.eos_token_id)
            elif self.args.tokenizer_type == 'LlamaTokenizer':
                doc_ids[-1].append(self.tokenizer.eos_token_id)
            else:
                doc_ids[-1].append(self.tokenizer.eos_token_id)
        ids[key] = doc_ids  # 将不同key对应的内容编码后存入ids
        return ids, len(json_line)  # 返回处理好序列id, len(json_line)用于后续显示处理进度


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, default=None,
                       help='Path to input JSON')
    group.add_argument('--json-keys', nargs='+', default=['text'],  # json中有用的key
                       help='space separate listed of keys to extract from json')
    group.add_argument('--split-sentences', action='store_true',  # 是否分句
                       help='Split documents into sentences.')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, required=True,  # tokenizer类型
                       choices=['BertWordPieceLowerCase','Llama3Tokenizer',
                                'QwenTokenizer'],
                       help='What type of tokenizer to use.')

    group.add_argument('--tokenizer_path', type=str, required=True)


    group.add_argument('--append-eod', default=True,
                       help='Append an <eod> token to the end of a document.')

    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=False,
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset-impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])


    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=40,  # 并行数
                       help='Number of worker processes to launch')
    group.add_argument('--log-interval', type=int, default=10000,  # 每隔多少行输出一次
                       help='Interval between progress updates')
    args = parser.parse_args() 

    return args


def preprocess_file(args):

    startup_start = time.time()

    print("Opening", args.input)
    fin = open(args.input, 'r', encoding='utf-8')



    encoder = Encoder(args)
    tokenizer = encoder.tokenizer

    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
    encoded_docs = pool.imap(encoder.encode, fin,args.workers)  # 对文件内容调用encoder中的encode方法
    level = "document"
    if args.split_sentences:
        level = "sentence"
    print(f"Output prefix: {args.output_prefix}")
    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    for key in args.json_keys:
        output_bin_files[key] = "{}_{}_{}.bin".format(args.output_prefix,  # 设置输出的文件名
                                                      key, level)
        output_idx_files[key] = "{}_{}_{}.idx".format(args.output_prefix,
                                                      key, level)
        builders[key] = indexed_dataset.make_builder(output_bin_files[key],  # MMapIndexedDatasetBuilder
                                                     impl=args.dataset_impl,
                                                     vocab_size=encoder.tokenizer.vocab_size)

    startup_end = time.time()  # 用于显示处理进度
    proc_start = time.time()
    total_bytes_processed = 0  # 总共已经处理过的bytes
    print("Time to startup:", startup_end - startup_start)

    for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
        # doc为encode后的全部文本，bytes_processed为文本长度，即encode函数的返回值
        total_bytes_processed += bytes_processed
        for key, sentences in doc.items():
            # 依次处理每个json_key对应的部分内容
            if len(sentences) == 0:
                continue
            for sentence in sentences:
                builders[key].add_item(torch.IntTensor(sentence))  # 使用builder将id化的sentence，加到.bin文件上去，
                # 并保存其中的token（word piece)的数量到self._sizes
                # （后续供构造.idx文件使用）
            builders[key].end_document()  # 添加size
        if i % args.log_interval == 0:  # 显示处理进度
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            print(args.input.split('/')[-1],
                  f"Processed {i:.2f} documents".format(i),
                  f"Conumed time:{elapsed:.2f}".format(elapsed),
                  f"({i / elapsed:.2f} docs/s, {mbs:.2f} MB/s).".format(i,elapsed,mbs),
                  file=sys.stderr)

    for key in args.json_keys:
        builders[key].finalize(output_idx_files[key])  # 负责构造.idx文件


def main():
    args = get_args()
    preprocess_file(args)

if __name__ == '__main__':
    main()
