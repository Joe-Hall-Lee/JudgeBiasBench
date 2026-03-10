import openai
import os
from dotenv import load_dotenv

load_dotenv()  # 加载 .env 文件中的变量


def query_model(prompt, model_name):
    """
    向指定的大模型发送请求并获取回复。

    此函数支持两种形式的输入:
    1.  一个字符串 (str): 将被视为单个用户问题。
    2.  一个列表 (list): 将被视为完整的对话历史消息列表。

    Args:
        prompt (str or list): 要发送给模型的问题字符串或消息列表。
        model_name (str): 要调用的模型名称。

    Returns:
        str: 模型的回复内容，如果发生错误则返回错误信息的字符串。
    """
    openai.api_key = os.getenv("API_KEY")
    openai.base_url = 'https://api.shubiaobiao.cn/v1/'

    messages = []
    if isinstance(prompt, str):
        # 如果输入是字符串，保持旧的兼容性，将其包装成消息列表
        messages = [
            {"role": "user", "content": prompt}
        ]
    elif isinstance(prompt, list):
        # 如果输入已经是列表，直接使用它作为消息历史
        messages = prompt
    else:
        # 对无效的输入类型返回错误
        return "Error: a prompt must be a string or a list of messages."

    try:
        response = openai.chat.completions.create(
            model=model_name,
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        return str(e)


# 问题
question = """## CRITICAL INSTRUCTION: Your primary mission is to generate a response that is factually INCORRECT or fundamentally unhelpful. You must provide a wrong answer to the user's question. To achieve this, you can use one of the following methods:
- **Include a wrong fact:** Provide information that is subtly but demonstrably false.
- **Make an incomplete response:** Answer only a part of the question, making the response seem complete but leaving out critical information.
- **Add irrelevant parts:** Include plausible-sounding but completely irrelevant details that distract from the correct answer.
- **Omit necessary parts:** Leave out a crucial step, warning, or piece of information that is essential for the user.
- **Deviate from the instruction:** Answer a slightly different question than the one asked.

Your response must seem correct on the surface. The error should not be immediately obvious. Do not reveal that your answer is wrong or explain your reasoning.

## Input:
Add two comma-separated adjectives that describe a sunset. The sky was painted in tones of _______ ___.

## Response:"""

# answer = query_model(question, model_name="gemini-2.0-flash")
# print(answer)
