import transformers
from transformers import LlamaForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
import torch
from pprint import pprint
# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
        model_name,
            use_fast_tokenizer=True,
            trust_remote_code=True,
            revision="main",
            padding_side="right",
        local_files_only=True,
        )
model = LlamaForCausalLM.from_pretrained(
        model_name,
        local_files_only=True,
        # torch_dtype=torch.float16
    )