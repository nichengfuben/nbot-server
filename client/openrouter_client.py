# openrouter_client.py
import asyncio
import aiohttp
import json
import base64
import mimetypes
import os
import time
from typing import AsyncGenerator, Optional, List, Union, Dict
from pathlib import Path

# 配置常量
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "x-ai/grok-4-fast:free"
TIMEOUT = 60
RECOVERY_TIME = 60  # 失败恢复时间（秒）

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.openrouter_accounts import *
from common.logger import get_logger
logger = get_logger("openrouter")

# 失败的密钥字典，记录失败时间
failed_keys: Dict[str, float] = {}

class OpenRouterClient:
    def __init__(self, max_concurrent: int = 5):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
    def _clean_expired_keys(self):
        """清理已过期的失败密钥"""
        global failed_keys
        current_time = time.time()
        expired_keys = []
        
        for key, fail_time in failed_keys.items():
            if current_time - fail_time >= RECOVERY_TIME:
                expired_keys.append(key)
        
        # 移除过期的密钥
        for key in expired_keys:
            del failed_keys[key]
            logger.info(f"密钥已恢复: {key[:20]}...")
    
    def get_available_key(self) -> Optional[str]:
        """获取可用的API密钥"""
        global failed_keys
        
        # 首先清理过期的失败密钥
        self._clean_expired_keys()
        
        # 获取当前可用的密钥
        available_keys = [key for key in API_KEYS if key not in failed_keys]
        
        if not available_keys:
            # 如果没有可用密钥，检查是否有即将恢复的密钥
            if failed_keys:
                current_time = time.time()
                # 找出最早失败的密钥和剩余等待时间
                earliest_key = min(failed_keys.items(), key=lambda x: x[1])
                wait_time = RECOVERY_TIME - (current_time - earliest_key[1])
                if wait_time > 0:
                    logger.info(f"所有密钥暂时不可用，最短需等待 {wait_time:.1f} 秒")
                return None
            else:
                logger.info("没有配置API密钥")
                return None
                
        return random.choice(available_keys)
    
    def _mark_key_failed(self, api_key: str):
        """标记密钥失败"""
        global failed_keys
        failed_keys[api_key] = time.time()
        logger.info(f"密钥暂时失败，将在{RECOVERY_TIME}秒后恢复: {api_key[:20]}...")
    
    def _prepare_image_content(self, image_path_or_url: Union[str, Path]) -> dict:
        """准备图片内容，支持本地文件和URL"""
        if isinstance(image_path_or_url, Path):
            image_path_or_url = str(image_path_or_url)
        
        if image_path_or_url.startswith(('http://', 'https://')):
            # URL图片
            return {
                "type": "image_url",
                "image_url": {
                    "url": image_path_or_url
                }
            }
        else:
            # 本地文件图片
            if not os.path.exists(image_path_or_url):
                raise FileNotFoundError(f"图片文件不存在: {image_path_or_url}")
            
            # 读取图片并编码为base64
            with open(image_path_or_url, 'rb') as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # 获取MIME类型
            mime_type, _ = mimetypes.guess_type(image_path_or_url)
            if not mime_type:
                mime_type = "image/jpeg"  # 默认类型
            
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{image_data}"
                }
            }
    
    async def chat_stream(self, 
                         prompt: str, 
                         image_paths: Optional[List[Union[str, Path]]] = None,
                         temperature: float = 0.2,
                         max_retries: int = 3) -> AsyncGenerator[str, None]:
        """流式聊天，支持多模态和自动重试"""
        retry_count = 0
        
        while retry_count < max_retries:
            api_key = self.get_available_key()
            if not api_key:
                # 如果没有可用密钥，等待一段时间后重试
                if retry_count < max_retries - 1:
                    await asyncio.sleep(5)
                    retry_count += 1
                    continue
                else:
                    raise RuntimeError("没有可用的API密钥")
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com",
                "X-Title": "OpenRouter Client"
            }
            
            # 构建消息内容
            content = [{"type": "text", "text": prompt}]
            
            # 添加图片内容
            if image_paths:
                for image_path in image_paths:
                    image_content = self._prepare_image_content(image_path)
                    content.append(image_content)
            
            body = {
                "model": MODEL_NAME,
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                "stream": True,
                "max_tokens": 4000,
                "temperature": temperature
            }
            
            async with self.semaphore:
                timeout = aiohttp.ClientTimeout(total=TIMEOUT)
                
                try:
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.post(OPENROUTER_URL, headers=headers, json=body) as response:
                            if response.status != 200:
                                error_text = await response.text()
                                logger.info(f"API错误 ({response.status}): {error_text}")
                                self._mark_key_failed(api_key)
                                
                                # 重试
                                if retry_count < max_retries - 1:
                                    retry_count += 1
                                    await asyncio.sleep(1)
                                    continue
                                else:
                                    raise RuntimeError(f"API调用失败: {error_text}")
                            
                            # 成功，开始流式输出
                            success = False
                            async for line in response.content:
                                line = line.decode("utf-8").strip()
                                if not line.startswith("data: "):
                                    continue
                                
                                data = line[6:]
                                if data == "[DONE]":
                                    success = True
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
                            
                            if success:
                                return
                                
                except asyncio.TimeoutError:
                    logger.info(f"请求超时")
                    self._mark_key_failed(api_key)
                    if retry_count < max_retries - 1:
                        retry_count += 1
                        await asyncio.sleep(1)
                        continue
                    else:
                        raise
                        
                except Exception as e:
                    logger.info(f"流式调用异常: {e}")
                    self._mark_key_failed(api_key)
                    if retry_count < max_retries - 1:
                        retry_count += 1
                        await asyncio.sleep(1)
                        continue
                    else:
                        raise
    
    async def chat(self, 
                  prompt: str, 
                  image_paths: Optional[List[Union[str, Path]]] = None,
                  temperature: float = 0.2,
                  max_retries: int = 3) -> str:
        """非流式聊天，支持多模态和自动重试"""
        retry_count = 0
        
        while retry_count < max_retries:
            api_key = self.get_available_key()
            if not api_key:
                # 如果没有可用密钥，等待一段时间后重试
                if retry_count < max_retries - 1:
                    await asyncio.sleep(5)
                    retry_count += 1
                    continue
                else:
                    raise RuntimeError("没有可用的API密钥")
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com",
                "X-Title": "OpenRouter Client"
            }
            
            # 构建消息内容
            content = [{"type": "text", "text": prompt}]
            
            # 添加图片内容
            if image_paths:
                for image_path in image_paths:
                    image_content = self._prepare_image_content(image_path)
                    content.append(image_content)
            
            body = {
                "model": MODEL_NAME,
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                "stream": False,
                "max_tokens": 4000,
                "temperature": temperature
            }
            
            async with self.semaphore:
                timeout = aiohttp.ClientTimeout(total=TIMEOUT)
                
                try:
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.post(OPENROUTER_URL, headers=headers, json=body) as response:
                            if response.status != 200:
                                error_text = await response.text()
                                logger.info(f"API错误 ({response.status}): {error_text}")
                                self._mark_key_failed(api_key)
                                
                                # 重试
                                if retry_count < max_retries - 1:
                                    retry_count += 1
                                    await asyncio.sleep(1)
                                    continue
                                else:
                                    raise RuntimeError(f"API调用失败: {error_text}")
                            
                            data = await response.json()
                            return data['choices'][0]['message']['content'].strip()
                            
                except asyncio.TimeoutError:
                    logger.info(f"请求超时")
                    self._mark_key_failed(api_key)
                    if retry_count < max_retries - 1:
                        retry_count += 1
                        await asyncio.sleep(1)
                        continue
                    else:
                        raise
                        
                except Exception as e:
                    logger.info(f"非流式调用异常: {e}")
                    self._mark_key_failed(api_key)
                    if retry_count < max_retries - 1:
                        retry_count += 1
                        await asyncio.sleep(1)
                        continue
                    else:
                        raise

# 便捷函数
async def quick_chat(prompt: str, 
                    image_paths: Optional[List[Union[str, Path]]] = None,
                    temperature: float = 0.2) -> str:
    """快速聊天（非流式）"""
    client = OpenRouterClient()
    return await client.chat(prompt, image_paths, temperature)

async def quick_chat_stream(prompt: str, 
                           image_paths: Optional[List[Union[str, Path]]] = None,
                           temperature: float = 0.2) -> AsyncGenerator[str, None]:
    """快速聊天（流式）"""
    client = OpenRouterClient()
    async for chunk in client.chat_stream(prompt, image_paths, temperature):
        yield chunk

# 获取失败密钥状态
def get_failed_keys_status() -> Dict[str, float]:
    """获取失败密钥的状态信息"""
    global failed_keys
    current_time = time.time()
    status = {}
    for key, fail_time in failed_keys.items():
        remaining = max(0, RECOVERY_TIME - (current_time - fail_time))
        status[key[:20] + "..."] = remaining
    return status

# 示例使用
async def main():
    client = OpenRouterClient()
    
    # 纯文本示例
    logger.info("=== 纯文本聊天 ===")
    try:
        response = await client.chat("你好，请介绍一下自己")
        logger.info(response)
    except Exception as e:
        logger.info(f"错误: {e}")
    
    # 查看失败密钥状态
    status = get_failed_keys_status()
    if status:
        logger.info("\n=== 失败密钥状态 ===")
        for key, remaining in status.items():
            logger.info(f"{key}: 剩余恢复时间 {remaining:.1f} 秒")
    
    # 多模态示例（URL图片）
    logger.info("\n=== 多模态聊天（URL图片）===")
    try:
        image_url = "https://ts1.cn.mm.bing.net/th/id/R-C.987f582c510be58755c4933cda68d525?rik=C0D21hJDYvXosw&riu=http%3a%2f%2fimg.pconline.com.cn%2fimages%2fupload%2fupc%2ftx%2fwallpaper%2f1305%2f16%2fc4%2f20990657_1368686545122.jpg&ehk=netN2qzcCVS4ALUQfDOwxAwFcy41oxC%2b0xTFvOYy5ds%3d&risl=&pid=ImgRaw&r=0"
        response = await client.chat("请用中文描述一下这张图片", [image_url])
        logger.info(response)
    except Exception as e:
        logger.info(f"错误: {e}")
    
    # 流式示例
    logger.info("\n=== 流式聊天 ===")
    try:
        async for chunk in client.chat_stream("写一首关于春天的短诗"):
            logger.info(chunk, end="", flush=True)
        logger.info("")  # 换行
    except Exception as e:
        logger.info(f"错误: {e}")

if __name__ == "__main__":
    import random
    asyncio.run(main())
