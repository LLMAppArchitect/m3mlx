from datetime import datetime
import time

from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
from mlx_lm import load, generate

# 定义FastAPI应用
app = FastAPI()

# https://huggingface.co/mlx-community/gemma-2-9b-it-8bit
# model, tokenizer = load("mlx-community/gemma-2-9b-it-8bit")
# https://huggingface.co/google/gemma-2-9b-it
model, tokenizer = load("google/gemma-2-9b-it")

# 定义输入数据的模型
class InputData(BaseModel):
    prompt: str
    max_tokens: int = 100
    verbose: bool = True


seg = "================================================================================================================================"


# 定义POST接口
@app.post("/generate")
def get_generation(input_data: InputData):
    try:
        print(seg)
        print(input_data.prompt)
        s = int(time.time())
        print("开始时间：", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        # 调用模型推理函数
        response = generate(
            model,
            tokenizer,
            prompt=input_data.prompt,
            verbose=True,
            max_tokens=input_data.max_tokens
        )

        # 打印结果
        # print(response)

        t = int(time.time())
        print("结束时间：", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print("耗时(s)：", t - s)
        print("总字数：", len(response))
        print(seg)

        # 返回生成结果
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 启动服务
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8123)
