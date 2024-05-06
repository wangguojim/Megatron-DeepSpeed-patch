
import transformers

def build_tokenizer(tokenizer_path,tokenizer_cls):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_path,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True,
        )
    if 'QwenTokenizer' in  tokenizer_cls:
        tokenizer.eos_token_id = 151643
        tokenizer.pad_token_id = 151644

    elif 'LlamaTokenizer' == tokenizer_cls:
        tokenizer.pad_token = '<unk>'
        tokenizer.eos_token = '</s>'

    elif 'Llama3Tokenizer' ==  tokenizer_cls:
        tokenizer.pad_token = tokenizer.bos_token

    else:
        raise NotImplementedError

    return tokenizer