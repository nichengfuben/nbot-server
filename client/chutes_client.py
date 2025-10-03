import asyncio
from common.logger import get_logger
logger = get_logger("chutes")
import aiohttp
import json
import time
import random
from typing import AsyncGenerator, Optional, Dict

# 配置常量
CHUTES_URL = "https://llm.chutes.ai/v1/chat/completions"
MODEL_NAME = "zai-org/GLM-4.5-Air"
TIMEOUT = 60
RECOVERY_TIME = 60  # 恢复时间（秒）

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.chutes_accounts import *

# 失败的密钥及其失败时间
failed_keys: Dict[str, float] = {}

class ChutesClient:
    def __init__(self, max_concurrent: int = 5):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
    def clean_expired_failures(self):
        """清理过期的失败记录"""
        global failed_keys
        current_time = time.time()
        expired_keys = [
            key for key, fail_time in failed_keys.items()
            if current_time - fail_time >= RECOVERY_TIME
        ]
        for key in expired_keys:
            logger.info(f"密钥已恢复: {key[:8]}...")
            del failed_keys[key]
    
    def get_available_key(self) -> Optional[str]:
        """获取可用的API密钥"""
        global failed_keys
        
        # 首先清理过期的失败记录
        self.clean_expired_failures()
        
        # 获取当前可用的密钥
        available_keys = [key for key in API_KEYS if key not in failed_keys]
        
        if not available_keys:
            # 如果没有可用密钥，检查是否有即将恢复的密钥
            if failed_keys:
                current_time = time.time()
                # 找出最早失败的密钥及其剩余恢复时间
                earliest_key = min(failed_keys.items(), key=lambda x: x[1])
                remaining_time = RECOVERY_TIME - (current_time - earliest_key[1])
                if remaining_time > 0:
                        logger.info(f"所有密钥暂时不可用，最快将在 {remaining_time:.1f} 秒后恢复")
                return None
            return None
            
        return random.choice(available_keys)
    
    def mark_key_failed(self, api_key: str):
        """标记密钥失败"""
        global failed_keys
        failed_keys[api_key] = time.time()
        logger.info(f"密钥失败，将在 {RECOVERY_TIME} 秒后恢复: {api_key[:8]}...")
    
    async def chat_stream(self, prompt: str, temperature: float = 0.2) -> AsyncGenerator[str, None]:
        """流式聊天"""
        api_key = self.get_available_key()
        if not api_key:
            raise RuntimeError("当前没有可用的API密钥，请稍后重试")
            
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        body = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "max_tokens": 10000,
            "temperature": temperature
        }
        
        async with self.semaphore:
            timeout = aiohttp.ClientTimeout(total=TIMEOUT)
            
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(CHUTES_URL, headers=headers, json=body) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.info(f"API错误 ({response.status}): {error_text}")
                            self.mark_key_failed(api_key)
                            raise RuntimeError(f"API调用失败: {error_text}")
                        
                        async for line in response.content:
                            line = line.decode("utf-8").strip()
                            if not line.startswith("data: "):
                                continue
                            
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            
                            try:
                                chunk = json.loads(data)
                                if chunk.get("choices"):
                                    delta = chunk["choices"][0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                continue
                                
            except Exception as e:
                logger.info(f"流式调用异常: {e}")
                self.mark_key_failed(api_key)
                raise
    
    async def chat(self, prompt: str, temperature: float = 0.2) -> str:
        """非流式聊天"""
        api_key = self.get_available_key()
        if not api_key:
            raise RuntimeError("当前没有可用的API密钥，请稍后重试")
            
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        body = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": 10000,
            "temperature": temperature
        }
        
        async with self.semaphore:
            timeout = aiohttp.ClientTimeout(total=TIMEOUT)
            
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(CHUTES_URL, headers=headers, json=body) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.info(f"API错误 ({response.status}): {error_text}")
                            self.mark_key_failed(api_key)
                            raise RuntimeError(f"API调用失败: {error_text}")
                        
                        data = await response.json()
                        return data['choices'][0]['message']['content'].strip()
                        
            except Exception as e:
                logger.info(f"非流式调用异常: {e}")
                self.mark_key_failed(api_key)
                raise

# 便捷函数
async def quick_chat(prompt: str, temperature: float = 0.2) -> str:
    """快速聊天（非流式）"""
    client = ChutesClient()
    return await client.chat(prompt, temperature)

async def quick_chat_stream(prompt: str, temperature: float = 0.2) -> AsyncGenerator[str, None]:
    """快速聊天（流式）"""
    client = ChutesClient()
    async for chunk in client.chat_stream(prompt, temperature):
        yield chunk

# 示例使用
async def main():
    client = ChutesClient()
    
    # 非流式示例
    logger.info("=== 非流式聊天 ===")
    try:
        response = await client.chat("你好，请介绍一下自己")
        logger.info(response)
    except Exception as e:
        logger.info(f"错误: {e}")
    
    logger.info("\n=== 流式聊天 ===")
    # 流式示例
    try:
        async for chunk in client.chat_stream("写一首关于春天的短诗"):
            logger.info(chunk, end="", flush=True)
        logger.info("")  # 换行
    except Exception as e:
        logger.info(f"错误: {e}")

if __name__ == "__main__":
    asyncio.run(main())
