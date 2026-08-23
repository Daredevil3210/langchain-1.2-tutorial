import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from rich import print as rprint
# 将env文件中的变量加载为环境变量
#override=True：表示.env优先
load_dotenv(override=True)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
model = ChatDeepSeek(
    api_key=DEEPSEEK_API_KEY,
    api_base=DEEPSEEK_BASE_URL,
    model_name="deepseek-v4-flash"
)
# 2、声明一个函数（工具）
def get_weather(city:str):
    return f"{city}天气晴朗"
#3、将函数绑定在模型上
model_with_tools=model.bind_tools([get_weather])
#4、调用模型
response=model_with_tools.invoke("北京的天气怎么样")
rprint(response)