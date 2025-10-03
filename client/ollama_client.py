# ollama_client.py
import asyncio
import time
from typing import AsyncGenerator, Optional, Dict, Any, List
from ollama import Client
from common.logger import get_logger
logger = get_logger("ollama")

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.ollama_accounts import *

class OllamaClient:
    global API_KEY
    def __init__(self, 
                 host: str = "https://ollama.com",
                 token: Optional[str] = API_KEY,
                 model: str = "gpt-oss:120b",
                 max_concurrent: int = 5):
        """
        初始化 Ollama 客户端
        
        Args:
            host: Ollama 服务器地址
            token: 认证token
            model: 使用的模型名称
            max_concurrent: 最大并发请求数
        """
        self.host = host
        self.token = token
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.last_call_time = 0
        
    def _create_client(self) -> Client:
        """创建 Ollama 客户端"""
        headers = {}
        if self.token:
            headers['Authorization'] = self.token
            
        return Client(host=self.host, headers=headers)
    
    def _log(self, message: str):
        """简单的日志输出"""
        logger.info(f"[OllamaClient] {message}")
    
    def _filter_think_tags(self, text: str) -> str:
        """过滤思考标签（可根据需要自定义）"""
        # 这里可以添加你的过滤逻辑
        return text
    
    async def chat_stream(self, 
                         prompt: str, 
                         temperature: float = 0.2,
                         system: Optional[str] = None) -> AsyncGenerator[str, None]:
        """
        流式聊天
        
        Args:
            prompt: 用户输入
            temperature: 温度参数
            system: 系统提示词
            
        Yields:
            逐个token的响应内容
        """
        async with self.semaphore:
            try:
                self._log("开始流式调用")
                start_time = time.time()
                received_first_token = False
                
                client = self._create_client()
                
                messages = []
                if system:
                    messages.append({'role': 'system', 'content': system})
                messages.append({'role': 'user', 'content': prompt})
                
                for part in client.chat(
                    self.model, 
                    messages=messages, 
                    stream=True,
                    options={'temperature': temperature}
                ):
                    content = part.get('message', {}).get('content', '')
                    if content:
                        if not received_first_token:
                            received_first_token = True
                            self._log(f"收到第一个token，耗时: {time.time()-start_time:.2f}s")
                        yield content
                
                if not received_first_token:
                    self._log(f"警告: 未收到任何有效token，总耗时: {time.time()-start_time:.2f}s")
                    raise Exception("未返回任何有效数据")
                    
            except Exception as e:
                self._log(f"流式调用失败: {e}")
                raise
    
    async def chat(self, 
                  prompt: str, 
                  temperature: float = 0.2,
                  system: Optional[str] = None) -> str:
        """
        非流式聊天
        
        Args:
            prompt: 用户输入
            temperature: 温度参数
            system: 系统提示词
            
        Returns:
            完整的响应内容
        """
        async with self.semaphore:
            try:
                self._log("开始非流式调用")
                
                client = self._create_client()
                
                messages = []
                if system:
                    messages.append({'role': 'system', 'content': system})
                messages.append({'role': 'user', 'content': prompt})
                
                response = client.chat(
                    self.model, 
                    messages=messages, 
                    stream=False,
                    options={'temperature': temperature}
                )
                
                result = response.get('message', {}).get('content', '')
                
                if result:
                    self._log("调用成功")
                    return self._filter_think_tags(result).strip()
                else:
                    raise Exception("返回空结果")
                    
            except Exception as e:
                self._log(f"调用失败: {e}")
                raise
    
    async def chat_with_history(self, 
                               messages: List[Dict[str, str]], 
                               temperature: float = 0.2) -> str:
        """
        带历史记录的聊天
        
        Args:
            messages: 消息历史列表
            temperature: 温度参数
            
        Returns:
            响应内容
        """
        async with self.semaphore:
            try:
                self._log("开始历史对话调用")
                
                client = self._create_client()
                
                response = client.chat(
                    self.model, 
                    messages=messages, 
                    stream=False,
                    options={'temperature': temperature}
                )
                
                result = response.get('message', {}).get('content', '')
                
                if result:
                    self._log("历史对话调用成功")
                    return self._filter_think_tags(result).strip()
                else:
                    raise Exception("返回空结果")
                    
            except Exception as e:
                self._log(f"历史对话调用失败: {e}")
                raise
    
    async def batch_chat(self, 
                        prompts: List[str], 
                        temperature: float = 0.2) -> List[str]:
        """
        批量聊天
        
        Args:
            prompts: 提示词列表
            temperature: 温度参数
            
        Returns:
            响应列表
        """
        tasks = [self.chat(prompt, temperature) for prompt in prompts]
        return await asyncio.gather(*tasks)

# 使用示例
async def main():
    client = OllamaClient()
    
    # 简单聊天
    response = await client.chat("你好，请介绍一下自己")
    logger.info("非流式响应:")
    logger.info(response)
    
    # 流式聊天
    logger.info("\n流式响应:")
    async for chunk in client.chat_stream("讲一个简短的故事"):
        logger.info(chunk, end="", flush=True)
    logger.info("")
    


if __name__ == "__main__":
    asyncio.run(main())
