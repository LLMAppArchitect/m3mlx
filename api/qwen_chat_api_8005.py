from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
from mlx_lm import load, generate

# 定义FastAPI应用
app = FastAPI()

# 加载模型: https://huggingface.co/Qwen/Qwen2-7B-Instruct-MLX
model, tokenizer = load("Qwen/Qwen2-7B-Instruct-MLX", tokenizer_config={"eos_token": "<|im_end|>"})

example_prompt = """

### 角色 Role ###
您是一位世界级人工智能专家,程序员,软件架构师,CTO,世界顶级技术畅销书作者，计算机图灵奖获得者，计算机领域大师。

### 任务目标 GOAL ###
现在请您以《【大模型应用开发动手做AI Agent】Plan-and-Solve策略的提出》为标题， 使用逻辑清晰、结构紧凑、简单易懂的专业的技术语言（章节标题要非常吸引读者），写一篇有深度有思考有见解的专业IT领域的技术博客文章。

### 约束条件 CONSTRAINTS ###
- 字数要求：文章字数一定要大于8000字。
- 尽最大努力给出核心概念原理和架构的 Mermaid 流程图(要求：Mermaid 流程节点中不要有括号、逗号等特殊字符)。
- 文章各个段落章节的子目录请具体细化到三级目录。
- 直接开始文章正文部分的撰写。
- 格式要求：文章内容使用markdown格式输出；文章中的数学公式请使用latex格式，latex嵌入文中独立段落使用 $$，段落内使用 $
- 完整性要求：文章内容必须要完整，不能只提供概要性的框架和部分内容，不要只是给出目录。不要只给概要性的框架和部分内容。
- 内容要求：文章核心章节内容必须包含如下目录内容(文章结构模板)：
--------------------------------

关键词：

## 1. 背景介绍
### 1.1  问题的由来
### 1.2  研究现状
### 1.3  研究意义
### 1.4  本文结构
## 2. 核心概念与联系
## 3. 核心算法原理 & 具体操作步骤
### 3.1  算法原理概述
### 3.2  算法步骤详解
### 3.3  算法优缺点
### 3.4  算法应用领域
## 4. 数学模型和公式 & 详细讲解 & 举例说明
### 4.1  数学模型构建
### 4.2  公式推导过程
### 4.3  案例分析与讲解
### 4.4  常见问题解答
## 5. 项目实践：代码实例和详细解释说明
### 5.1  开发环境搭建
### 5.2  源代码详细实现
### 5.3  代码解读与分析
### 5.4  运行结果展示
## 6. 实际应用场景
### 6.4  未来应用展望
## 7. 工具和资源推荐
### 7.1  学习资源推荐
### 7.2  开发工具推荐
### 7.3  相关论文推荐
### 7.4  其他资源推荐
## 8. 总结：未来发展趋势与挑战
### 8.1  研究成果总结
### 8.2  未来发展趋势
### 8.3  面临的挑战
### 8.4  研究展望
## 9. 附录：常见问题与解答

--------------------------------

!!!Important:必须要严格遵循上面"约束条件 CONSTRAINTS"中的所有要求撰写这篇文章!!!

### 文章正文内容部分 Content ###
现在，请开始撰写文章正文部分：

# 【大模型应用开发动手做AI Agent】Plan-and-Solve策略的提出

作者：禅与计算机程序设计艺术 / Zen and the Art of Computer Programming 

"""


# 定义输入数据的模型
class InputData(BaseModel):
    prompt: str
    max_tokens: int = 100
    verbose: bool = True


# 定义POST接口
@app.post("/generate")
def get_generation(input_data: InputData):
    with open("/Users/bytedance/ai/m3mlx/example_blog.md", 'r') as f:
        example_blog = f.read()

    messages = [
        {"role": "user", "content": example_prompt},
        {"role": "assistant", "content": example_blog},
        {"role": "user", "content": input_data.prompt}
    ]

    print(messages)

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
    uvicorn.run(app, host="0.0.0.0", port=8005)
