# 安装

```bash
conda create -n m3mlx python=3.11
conda activate m3mlx

pip install mlx-lm  fastapi uvicorn sentencepiece
```

# 启动服务

```
python phi3_api.py
python yi_chat_api.py
python qwen_chat_api.py
python openchat_api.py
```

# Test

```bash
curl -X POST "http://127.0.0.1:8000/generate" -H "Content-Type: application/json" -d '{"prompt": "使用 golang 实现一个 DAG task scheduler 源代码，并给出详细注释说明", "max_tokens":12800, "verbose":true}'

```

# 模型列表

## MLX: An array framework for Apple silicon:
https://github.com/LLMAppArchitect/mlx
https://ml-explore.github.io/mlx/build/html/index.html
[https://huggingface.co/mlx-community](https://huggingface.co/mlx-community)

MLX is a NumPy-like array framework designed for efficient and flexible machine learning on Apple silicon, brought to you by Apple machine learning research.
MLX 是一个类似 NumPy 的数组框架，专为 Apple 芯片上高效灵活的机器学习而设计，由 Apple 机器学习研究团队为您带来。

The Python API closely follows NumPy with a few exceptions. MLX also has a fully featured C++ API which closely follows the Python API.
Python API 紧密遵循 NumPy，但有一些例外。 MLX 还拥有功能齐全的 C++ API，该 API 紧密遵循 Python API。

The main differences between MLX and NumPy are:
MLX 和 NumPy 之间的主要区别是：

Composable function transformations: MLX has composable function transformations for automatic differentiation, automatic vectorization, and computation graph optimization.
可组合函数转换：MLX 具有用于自动微分、自动矢量化和计算图优化的可组合函数转换。

Lazy computation: Computations in MLX are lazy. Arrays are only materialized when needed.
惰性计算：MLX 中的计算是惰性的。数组仅在需要时才会具体化。

Multi-device: Operations can run on any of the supported devices (CPU, GPU, …)
多设备：操作可以在任何支持的设备上运行（CPU、GPU……）

The design of MLX is inspired by frameworks like PyTorch, Jax, and ArrayFire. A notable difference from these frameworks and MLX is the unified memory model. Arrays in MLX live in shared memory. Operations on MLX arrays can be performed on any of the supported device types without performing data copies. Currently supported device types are the CPU and GPU.
MLX 的设计灵感来自 PyTorch、Jax 和 ArrayFire 等框架。这些框架和 MLX 的显着区别是统一内存模型。 MLX 中的数组位于共享内存中。可以在任何支持的设备类型上执行 MLX 阵列上的操作，而无需执行数据复制。目前支持的设备类型是 CPU 和 GPU。


## Supported Models

[https://pypi.org/project/mlx-lm/](https://pypi.org/project/mlx-lm/)


The example supports Hugging Face format Mistral, Llama, and Phi-2 style models. If the model you want to run is not supported, file an issue or better yet, submit a pull request.

Here are a few examples of Hugging Face models that work with this example:

```
mistralai/Mistral-7B-v0.1
meta-llama/Llama-2-7b-hf
deepseek-ai/deepseek-coder-6.7b-instruct
01-ai/Yi-6B-Chat
microsoft/phi-2
mistralai/Mixtral-8x7B-Instruct-v0.1
Qwen/Qwen-7B
pfnet/plamo-13b
pfnet/plamo-13b-instruct
stabilityai/stablelm-2-zephyr-1_6b
```

Most Mistral, Llama, Phi-2, and Mixtral style models should work out of the box.

For some models (such as Qwen and plamo) the tokenizer requires you to enable the trust_remote_code option. You can do this by passing --trust-remote-code in the command line. If you don't specify the flag explicitly, you will be prompted to trust remote code in the terminal when running the model.

For Qwen models you must also specify the eos_token. You can do this by passing --eos-token "<|endoftext|>" in the command line.

These options can also be set in the Python API. For example:

```python
model, tokenizer = load(
    "qwen/Qwen-7B",
    tokenizer_config={"eos_token": "<|endoftext|>", "trust_remote_code": True},
)

```



## mlx-community/Phi-3-medium-128k-instruct-4bit

https://huggingface.co/mlx-community/Phi-3-medium-128k-instruct-4bit

---
language:
- multilingual
license: mit
tags:
- nlp
- code
- mlx
license_link: https://huggingface.co/microsoft/Phi-3-medium-128k-instruct/resolve/main/LICENSE
pipeline_tag: text-generation
inference:
  parameters:
    temperature: 0.7
widget:
- messages:
  - role: user
    content: Can you provide ways to eat combinations of bananas and dragonfruits?
---

# mlx-community/Phi-3-medium-128k-instruct-4bit

The Model [mlx-community/Phi-3-medium-128k-instruct-4bit](https://huggingface.co/mlx-community/Phi-3-medium-128k-instruct-4bit) was converted to MLX format from [microsoft/Phi-3-medium-128k-instruct](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct) using mlx-lm version **0.13.1**.

## Use with mlx

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Phi-3-medium-128k-instruct-4bit")
response = generate(model, tokenizer, prompt="hello", verbose=True)
```
## Conda environment setup

## Installation

用FastAPI增强为HTTP POST接口
首先，确保你已经安装了FastAPI和Uvicorn:
```bash
pip install fastapi uvicorn 
```

## Code
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mlx_lm import load, generate

# 定义FastAPI应用
app = FastAPI()

# 加载模型
model, tokenizer = load("mlx-community/Phi-3-medium-128k-instruct-4bit")

# 定义输入数据的模型
class InputData(BaseModel):
    prompt: str
    max_tokens: int = 10000
    verbose: bool = False

# 定义POST接口
@app.post("/generate")
def get_generation(input_data: InputData):
    try:
        # 调用模型推理函数
        response = generate(
            model,
            tokenizer,
            prompt=input_data.prompt,
            max_tokens=input_data.max_tokens,
            verbose=input_data.verbose
        )
        
        # 返回生成结果
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 启动服务
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

```


---

# FAQ

## Yi Chat

```bash 
Traceback (most recent call last):100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5.33G/5.33G [16:56<00:00, 2.96MB/s]
  File "/Users/bytedance/ai/phi3mlx/yi_chat_api.py", line 10, in <module>█████████████████████████████▌                                                               | 2.67G/5.35G [16:55<22:10, 2.01MB/s]
    model, tokenizer = load("mlx-community/Yi-1.5-34B-Chat-4bit")█████████████████████████████████████████████████████████████████████████████████████████████████████| 5.35G/5.35G [23:51<00:00, 5.12MB/s]
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/phi3mlx/lib/python3.11/site-packages/mlx_lm/utils.py", line 423, in load
    tokenizer = load_tokenizer(model_path, tokenizer_config)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/phi3mlx/lib/python3.11/site-packages/mlx_lm/tokenizer_utils.py", line 328, in load_tokenizer
    AutoTokenizer.from_pretrained(model_path, **tokenizer_config_extra),
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/phi3mlx/lib/python3.11/site-packages/transformers/models/auto/tokenization_auto.py", line 880, in from_pretrained
    return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/phi3mlx/lib/python3.11/site-packages/transformers/tokenization_utils_base.py", line 2110, in from_pretrained
    return cls._from_pretrained(
           ^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/phi3mlx/lib/python3.11/site-packages/transformers/tokenization_utils_base.py", line 2336, in _from_pretrained
    tokenizer = cls(*init_inputs, **init_kwargs)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/phi3mlx/lib/python3.11/site-packages/transformers/models/llama/tokenization_llama_fast.py", line 156, in __init__
    super().__init__(
  File "/opt/miniconda3/envs/phi3mlx/lib/python3.11/site-packages/transformers/tokenization_utils_fast.py", line 105, in __init__
    raise ValueError(
ValueError: Cannot instantiate this tokenizer from a slow version. If it's based on sentencepiece, make sure you have sentencepiece installed.

```

Install pip install sentencepiece:
```bash 
pip install sentencepiece
```
