from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
from mlx_lm import load, generate

# 定义FastAPI应用
app = FastAPI()

# 加载模型: https://huggingface.co/Qwen/Qwen2-7B-Instruct-MLX
model, tokenizer = load("Qwen/Qwen2-7B-Instruct-MLX", tokenizer_config={"eos_token": "<|im_end|>"})


# 定义输入数据的模型
class InputData(BaseModel):
    prompt: str
    max_tokens: int = 100
    verbose: bool = True


# 定义POST接口
@app.post("/generate")
def get_generation(input_data: InputData):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": input_data.prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    try:
        # 调用模型推理函数
        response = generate(
            model, tokenizer,
            prompt=text,
            verbose=True,
            top_p=0.8,
            temp=0.7,
            repetition_penalty=1.05,
            max_tokens=input_data.max_tokens
        )

        # 打印结果
        print(response)

        # 返回生成结果
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 启动服务
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
