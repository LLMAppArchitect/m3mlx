from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
from mlx_lm import load, generate

# 定义FastAPI应用
app = FastAPI()

# 加载模型: https://huggingface.co/mlx-community/Yi-1.5-34B-Chat-4bit
model, tokenizer = load("mlx-community/Yi-1.5-34B-Chat-4bit")


# 定义输入数据的模型
class InputData(BaseModel):
    prompt: str
    max_tokens: int = 100
    verbose: bool = True


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
            verbose=True,
            # repetition_penalty 是一个用于控制生成文本时重复词汇的惩罚因子。它可以帮助生成更具多样性和连贯性的文本。
            # 在设置 repetition_penalty 时，通常的取值范围在 1.0 到 2.0 之间。
            # 具体的取值需要根据实际应用场景和模型的表现进行调整。
            # 以下是一些常见的设置和其影响：
            # 1.0: default, 没有惩罚，模型生成的文本可能会有更多的重复词汇。
            # 1.1 - 1.5: 适度的惩罚，可以减少一些重复，但仍然保持生成文本的连贯性。
            # 1.5 - 2.0: 较强的惩罚，可以显著减少重复词汇，但可能会影响生成文本的流畅性。
            # 通常，建议从 1.1 开始尝试，并根据生成结果逐步调整。
            repetition_penalty=1.1,

            # repetition_context_size (int, optional):
            # The number of tokens to consider for repetition penalty (default 20).
            repetition_context_size=100,
        )

        # 打印结果
        print(response)

        # 返回生成结果
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 启动服务
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
