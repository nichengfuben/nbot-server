# minimax_client.py
import requests
import json
import base64
import os
from typing import Generator, AsyncGenerator, Optional
import aiohttp
import asyncio
import aiofiles
from common.logger import get_logger
logger = get_logger("minimax")

# 配置信息
BASE_URL = "https://ai.airoe.cn/v1/chat/completions"
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.minimax_accounts import *

MODEL = "minimax"
TEMPERATURE = 0.7

class AIROEAPIError(Exception):
    """AIROE API 专用异常类"""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"{status_code}: {message}")

def _is_url(path: str) -> bool:
    """判断是否为URL"""
    return path.startswith(('http://', 'https://'))

def _get_image_type(image_path: str) -> str:
    """根据文件扩展名推断图片类型"""
    if image_path.lower().endswith('.png'):
        return 'image/png'
    elif image_path.lower().endswith(('.jpg', '.jpeg')):
        return 'image/jpeg'
    elif image_path.lower().endswith('.gif'):
        return 'image/gif'
    elif image_path.lower().endswith('.webp'):
        return 'image/webp'
    else:
        return 'image/png'  # 默认类型

async def _process_image_async(image_path: str) -> dict:
    """异步处理图片，返回图片消息内容"""
    if _is_url(image_path):
        # URL 图片
        return {
            "type": "image_url",
            "image_url": {"url": image_path}
        }
    else:
        # 本地图片文件
        try:
            async with aiofiles.open(image_path, "rb") as image_file:
                image_data = await image_file.read()
                base64_image = base64.b64encode(image_data).decode('utf-8')
                image_type = _get_image_type(image_path)
                return {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image_type};base64,{base64_image}"
                    }
                }
        except FileNotFoundError:
            raise AIROEAPIError(0, f"找不到图片文件: {image_path}")
        except Exception as e:
            raise AIROEAPIError(0, f"读取图片文件时出错: {e}")

def _process_image_sync(image_path: str) -> dict:
    """同步处理图片，返回图片消息内容"""
    if _is_url(image_path):
        # URL 图片
        return {
            "type": "image_url",
            "image_url": {"url": image_path}
        }
    else:
        # 本地图片文件
        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                base64_image = base64.b64encode(image_data).decode('utf-8')
                image_type = _get_image_type(image_path)
                return {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image_type};base64,{base64_image}"
                    }
                }
        except FileNotFoundError:
            raise AIROEAPIError(0, f"找不到图片文件: {image_path}")
        except Exception as e:
            raise AIROEAPIError(0, f"读取图片文件时出错: {e}")

async def _build_message_content_async(question: str, image_path: Optional[str] = None) -> list:
    """异步构建消息内容"""
    if image_path:
        # 包含图片的消息
        content = [
            {"type": "text", "text": question},
            await _process_image_async(image_path)
        ]
    else:
        # 纯文本消息
        content = question
    return content

def _build_message_content_sync(question: str, image_path: Optional[str] = None):
    """同步构建消息内容"""
    if image_path:
        # 包含图片的消息
        content = [
            {"type": "text", "text": question},
            _process_image_sync(image_path)
        ]
    else:
        # 纯文本消息
        content = question
    return content

async def chat_non_stream(question: str, image_path: Optional[str] = None) -> str:
    """
    非流式请求：发送问题并返回完整回复文本
    
    Args:
        question: 用户问题
        image_path: 图片路径（可选），支持本地路径或URL
    """
    question = 3 * question
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    content = await _build_message_content_async(question, image_path)
    
    payload = {
        "model": MODEL,
        "group": "default",
        "messages": [
            {"role": "user", "content": content}
        ],
        "temperature": TEMPERATURE,
        "stream": False,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(BASE_URL, headers=headers, json=payload) as response:
            # 先读取响应内容
            response_text = await response.text()
            # 检查HTTP状态码并抛出异常
            if response.status == 401:
                raise AIROEAPIError(401, "API密钥无效或已过期")
            elif response.status == 403:
                raise AIROEAPIError(403, "访问被拒绝")
            elif response.status == 404:
                raise AIROEAPIError(404, "API端点不存在")
            elif response.status == 429:
                raise AIROEAPIError(429, "请求频率超限")
            elif response.status >= 500:
                raise AIROEAPIError(response.status, f"服务器错误: {response_text[:200]}")
            elif response.status != 200:
                raise AIROEAPIError(response.status, f"HTTP {response.status}: {response_text[:200]}")
            
            # 解析JSON
            try:
                data = json.loads(response_text)
            except json.JSONDecodeError:
                raise AIROEAPIError(0, f"JSON解析失败: {response_text[:200]}")
            
            # 检查API返回的错误
            if "error" in data:
                error_msg = data["error"].get("message", str(data["error"]))
                error_code = data["error"].get("code", "unknown")
                raise AIROEAPIError(response.status, f"{error_code}: {error_msg}")
            
            # 提取模型回复内容
            if "choices" not in data or len(data["choices"]) == 0:
                raise AIROEAPIError(response.status, "API返回格式错误：缺少choices字段")
            
            content = data["choices"][0]["message"]["content"]
            return content.strip()

async def chat_stream(question: str, image_path: Optional[str] = None) -> AsyncGenerator[str, None]:
    """
    流式请求：返回生成器，逐步输出回复内容
    
    Args:
        question: 用户问题
        image_path: 图片路径（可选），支持本地路径或URL
    """
    question = 3 * question
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    content = await _build_message_content_async(question, image_path)
    
    payload = {
        "model": MODEL,
        "group": "default",
        "messages": [
            {"role": "user", "content": content}
        ],
        "temperature": TEMPERATURE,
        "stream": True,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(BASE_URL, headers=headers, json=payload) as response:
            # 立即检查HTTP状态码
            if response.status == 401:
                raise AIROEAPIError(401, "API密钥无效或已过期")
            elif response.status == 403:
                raise AIROEAPIError(403, "访问被拒绝")
            elif response.status == 404:
                raise AIROEAPIError(404, "API端点不存在")
            elif response.status == 429:
                raise AIROEAPIError(429, "请求频率超限")
            elif response.status >= 500:
                error_text = await response.text()
                raise AIROEAPIError(response.status, f"服务器错误: {error_text[:200]}")
            elif response.status != 200:
                error_text = await response.text()
                raise AIROEAPIError(response.status, f"HTTP {response.status}: {error_text[:200]}")
            
            received_any_data = False
            buffer = ""
            
            async for line in response.content:
                if line:
                    received_any_data = True
                    line_str = line.decode("utf-8").strip()
                    buffer += line_str + "\n"
                    # 处理可能的多行数据
                    while "\ndata: " in buffer or buffer.startswith("data: "):
                        if buffer.startswith("data: "):
                            end_index = buffer.find("\n")
                            if end_index == -1:
                                break  # 等待更多数据
                            current_line = buffer[:end_index]
                            buffer = buffer[end_index + 1:]
                        else:
                            start_index = buffer.find("\ndata: ")
                            if start_index == -1:
                                break
                            # 处理 data: 之前的部分
                            buffer = buffer[start_index + 1:]
                            continue
                        
                        if current_line.startswith("data: "):
                            data_str = current_line[6:]  # 去掉 "data: "
                            
                            if data_str.strip() == "[DONE]":
                                return  # 正常结束
                            
                            if data_str.strip():  # 忽略空数据
                                try:
                                    data = json.loads(data_str)
                                    
                                    # 检查是否有错误
                                    if "error" in data:
                                        error_msg = data["error"].get("message", str(data["error"]))
                                        error_code = data["error"].get("code", "unknown")
                                        raise AIROEAPIError(response.status, f"{error_code}: {error_msg}")
                                    
                                    # 提取内容
                                    if "choices" in data and len(data["choices"]) > 0:
                                        delta = data["choices"][0].get("delta", {})
                                        if "content" in delta:
                                            if "undefined" == delta["content"]:
                                                raise AIROEAPIError(response.status, f"undefined错误")
                                            else:
                                                yield delta["content"]
                                            
                                except json.JSONDecodeError:
                                    # 忽略无法解析的行
                                    pass
            
            if not received_any_data:
                raise AIROEAPIError(response.status, "未收到任何流式数据")

# 同步版本函数（保持兼容性）
def chat_non_stream_sync(question: str, image_path: Optional[str] = None) -> str:
    """
    非流式请求（同步版本）
    
    Args:
        question: 用户问题
        image_path: 图片路径（可选），支持本地路径或URL
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    content = _build_message_content_sync(question, image_path)
    
    payload = {
        "model": MODEL,
        "group": "default",
        "messages": [
            {"role": "user", "content": content}
        ],
        "temperature": TEMPERATURE,
        "stream": False,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }

    response = requests.post(BASE_URL, headers=headers, json=payload)
    
    # 检查HTTP状态码
    if response.status_code == 401:
        raise AIROEAPIError(401, "API密钥无效或已过期")
    elif response.status_code == 403:
        raise AIROEAPIError(403, "访问被拒绝")
    elif response.status_code == 404:
        raise AIROEAPIError(404, "API端点不存在")
    elif response.status_code == 429:
        raise AIROEAPIError(429, "请求频率超限")
    elif response.status_code >= 500:
        raise AIROEAPIError(response.status_code, f"服务器错误: {response.text[:200]}")
    elif response.status_code != 200:
        raise AIROEAPIError(response.status_code, f"HTTP {response.status_code}: {response.text[:200]}")
    
    data = response.json()
    
    # 检查API返回的错误
    if "error" in data:
        error_msg = data["error"].get("message", str(data["error"]))
        error_code = data["error"].get("code", "unknown")
        raise AIROEAPIError(response.status_code, f"{error_code}: {error_msg}")
    
    # 提取模型回复内容
    if "choices" not in data or len(data["choices"]) == 0:
        raise AIROEAPIError(response.status_code, "API返回格式错误：缺少choices字段")
    
    content = data["choices"][0]["message"]["content"]
    return content.strip()

def chat_stream_sync(question: str, image_path: Optional[str] = None) -> Generator[str, None, None]:
    """
    流式请求（同步版本）
    
    Args:
        question: 用户问题
        image_path: 图片路径（可选），支持本地路径或URL
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    content = _build_message_content_sync(question, image_path)
    
    payload = {
        "model": MODEL,
        "group": "default",
        "messages": [
            {"role": "user", "content": content}
        ],
        "temperature": TEMPERATURE,
        "stream": True,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }

    with requests.post(BASE_URL, headers=headers, json=payload, stream=True) as response:
        # 检查HTTP状态码
        if response.status_code == 401:
            raise AIROEAPIError(401, "API密钥无效或已过期")
        elif response.status_code == 403:
            raise AIROEAPIError(403, "访问被拒绝")
        elif response.status_code == 404:
            raise AIROEAPIError(404, "API端点不存在")
        elif response.status_code == 429:
            raise AIROEAPIError(429, "请求频率超限")
        elif response.status_code >= 500:
            raise AIROEAPIError(response.status_code, f"服务器错误")
        elif response.status_code != 200:
            raise AIROEAPIError(response.status_code, f"HTTP {response.status_code}")
        
        for line in response.iter_lines(decode_unicode=True):
            if line:
                line = line.strip()
                if line.startswith("data: "):
                    data_str = line[6:]  # 去掉 "data: "
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        
                        # 检查是否有错误
                        if "error" in data:
                            error_msg = data["error"].get("message", str(data["error"]))
                            error_code = data["error"].get("code", "unknown")
                            raise AIROEAPIError(response.status_code, f"{error_code}: {error_msg}")
                        
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                    except json.JSONDecodeError:
                        continue

# 快捷函数：图片分析
async def analyze_image(image_path: str, question: str = "详细描述图像内容，在信息密度最大化下保持信息长度最小化，禁止使用换行。描述包括场景、物体、颜色、布局、细节特征及可能的含义。") -> str:
    """
    快捷图片分析函数
    
    Args:
        image_path: 图片路径（本地路径或URL）
        question: 分析问题（默认为详细描述）
    """
    return await chat_non_stream(question, image_path)

def analyze_image_sync(image_path: str, question: str = "详细描述图像内容，在信息密度最大化下保持信息长度最小化，禁止使用换行。描述包括场景、物体、颜色、布局、细节特征及可能的含义。") -> str:
    """
    快捷图片分析函数（同步版本）
    
    Args:
        image_path: 图片路径（本地路径或URL）
        question: 分析问题（默认为详细描述）
    """
    return chat_non_stream_sync(question, image_path)

if __name__ == "__main__":
    # 测试代码
    async def test_async():
        # 测试纯文本聊天
        logger.info("🔹 测试纯文本聊天:")
        try:
            result = await chat_non_stream("请写一首关于春天的诗")
            logger.info(result)
        except AIROEAPIError as e:
            logger.info(f"错误 [{e.status_code}]: {e.message}")
        
        # 测试图片分析（本地文件）
        image_path = r"C:\Users\dell\Pictures\Screenshots\屏幕截图 2025-07-29 083414.png"
        if os.path.exists(image_path):
            logger.info("\n🔹 测试本地图片分析:")
            try:
                result = await analyze_image(image_path)
                logger.info(result)
            except AIROEAPIError as e:
                logger.info(f"错误 [{e.status_code}]: {e.message}")
        
        # 测试图片聊天（流式）
        logger.info("\n🔹 测试图片流式聊天:")
        try:
            async for chunk in chat_stream("这张图片是什么？", image_path if os.path.exists(image_path) else None):
                logger.info(chunk, end="", flush=True)
            logger.info("")
        except AIROEAPIError as e:
            logger.info(f"错误 [{e.status_code}]: {e.message}")
    
    asyncio.run(test_async())
