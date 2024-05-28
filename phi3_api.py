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
            verbose=input_data.verbose,
            temp=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            repetition_context_size=1024,
        )
        # 打印结果
        print(response)

        # 返回生成结果
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 启动服务
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
