# cerebras_client.py
import asyncio
import random
from typing import AsyncGenerator, Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor
from cerebras.cloud.sdk import Cerebras
import logging
import sys
import os
import time
from datetime import datetime, timedelta
from common.logger import get_logger
app_logger = get_logger("cerebras")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.cerebras_accounts import *

# 配置常量
MODEL_NAME = "qwen-3-coder-480b"
MAX_COMPLETION_TOKENS = 65536
RECOVERY_TIME = 60  

# 失败的密钥字典（key -> 失败时间戳）
failed_keys: Dict[str, float] = {}

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CerebrasClient:
    def __init__(self, max_concurrent: int = 5):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        
    def get_available_key(self) -> Optional[str]:
        """获取可用的API密钥"""
        global failed_keys
        current_time = time.time()
        
        # 检查并移除已恢复的密钥
        keys_to_remove = []
        for key, fail_time in failed_keys.items():
            if current_time - fail_time >= RECOVERY_TIME:
                keys_to_remove.append(key)
        
        # 移除已恢复的密钥
        for key in keys_to_remove:
            del failed_keys[key]
            logger.info(f"密钥已恢复: {key[:8]}...")
        
        # 获取当前可用的密钥
        available_keys = [key for key in API_KEYS if key not in failed_keys]
        
        if not available_keys:
            # 如果没有可用密钥，检查是否有即将恢复的密钥
            if failed_keys:
                min_recovery_time = min(fail_time + RECOVERY_TIME - current_time 
                                       for fail_time in failed_keys.values())
                logger.warning(f"所有密钥暂时不可用，最快将在 {min_recovery_time:.1f} 秒后恢复")
            return None
            
        return random.choice(available_keys)
    
    def _mark_key_failed(self, api_key: str):
        """标记密钥失败"""
        global failed_keys
        failed_keys[api_key] = time.time()
        logger.warning(f"密钥失败: {api_key[:8]}...，将在 {RECOVERY_TIME} 秒后恢复")
    
    def _create_client(self, api_key: str) -> Cerebras:
        """创建Cerebras客户端"""
        return Cerebras(api_key=api_key)
    
    def _sync_chat_stream(self, api_key: str, prompt: str, 
                          temperature: float, top_p: float) -> list:
        """同步流式聊天（内部使用）"""
        client = self._create_client(api_key)
        chunks = []
        
        try:
            stream = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=MODEL_NAME,
                stream=True,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                temperature=temperature,
                top_p=top_p
            )
            
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    chunks.append(chunk.choices[0].delta.content)
                    
        except Exception as e:
            logger.error(f"流式调用失败: {e}")
            self._mark_key_failed(api_key)
            raise
            
        return chunks
    
    def _sync_chat(self, api_key: str, prompt: str, 
                   temperature: float, top_p: float) -> str:
        """同步非流式聊天（内部使用）"""
        client = self._create_client(api_key)
        
        try:
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=MODEL_NAME,
                stream=False,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                temperature=temperature,
                top_p=top_p
            )
            
            if response.choices and response.choices[0].message:
                return response.choices[0].message.content
            return ""
            
        except Exception as e:
            logger.error(f"非流式调用失败: {e}")
            self._mark_key_failed(api_key)
            raise
    
    async def chat_stream(self, 
                         prompt: str,
                         temperature: float = 0.7,
                         top_p: float = 0.8,
                         max_retries: int = 3) -> AsyncGenerator[str, None]:
        """异步流式聊天（支持重试）"""
        for attempt in range(max_retries):
            api_key = self.get_available_key()
            if not api_key:
                if attempt < max_retries - 1:
                    # 等待一段时间后重试
                    await asyncio.sleep(min(5, RECOVERY_TIME / 10))
                    continue
                raise RuntimeError("没有可用的API密钥")
            
            async with self.semaphore:
                loop = asyncio.get_event_loop()
                
                try:
                    # 在线程池中运行同步流式聊天
                    chunks = await loop.run_in_executor(
                        self.executor,
                        self._sync_chat_stream,
                        api_key,
                        prompt,
                        temperature,
                        top_p
                    )
                    
                    # 逐个yield返回内容
                    for chunk in chunks:
                        yield chunk
                    return  # 成功完成，退出重试循环
                    
                except Exception as e:
                    logger.error(f"异步流式调用异常（尝试 {attempt + 1}/{max_retries}）: {e}")
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(1)  # 短暂等待后重试
    
    async def chat(self, 
                  prompt: str,
                  temperature: float = 0.7,
                  top_p: float = 0.8,
                  max_retries: int = 3) -> str:
        """异步非流式聊天（支持重试）"""
        for attempt in range(max_retries):
            api_key = self.get_available_key()
            if not api_key:
                if attempt < max_retries - 1:
                    # 等待一段时间后重试
                    await asyncio.sleep(min(5, RECOVERY_TIME / 10))
                    continue
                raise RuntimeError("没有可用的API密钥")
            
            async with self.semaphore:
                loop = asyncio.get_event_loop()
                
                try:
                    # 在线程池中运行同步非流式聊天
                    response = await loop.run_in_executor(
                        self.executor,
                        self._sync_chat,
                        api_key,
                        prompt,
                        temperature,
                        top_p
                    )
                    return response
                    
                except Exception as e:
                    logger.error(f"异步非流式调用异常（尝试 {attempt + 1}/{max_retries}）: {e}")
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(1)  # 短暂等待后重试
    
    async def batch_chat(self, 
                        prompts: List[str],
                        temperature: float = 0.7,
                        top_p: float = 0.8) -> List[str]:
        """批量异步聊天"""
        tasks = [
            self.chat(prompt, temperature, top_p) 
            for prompt in prompts
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_status(self) -> dict:
        """获取客户端状态"""
        global failed_keys
        current_time = time.time()
        
        # 计算各密钥的恢复时间
        recovery_info = {}
        for key, fail_time in failed_keys.items():
            remaining_time = max(0, RECOVERY_TIME - (current_time - fail_time))
            recovery_info[key[:8] + "..."] = {
                "failed_at": datetime.fromtimestamp(fail_time).strftime("%Y-%m-%d %H:%M:%S"),
                "recovery_in": f"{remaining_time:.1f}秒"
            }
        
        return {
            "total_keys": len(API_KEYS),
            "available_keys": len([k for k in API_KEYS if k not in failed_keys]),
            "failed_keys": len(failed_keys),
            "recovery_info": recovery_info
        }
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

# 便捷函数
async def quick_chat(prompt: str, 
                    temperature: float = 0.7,
                    top_p: float = 0.8) -> str:
    """快速聊天（非流式）"""
    client = CerebrasClient()
    return await client.chat(prompt, temperature, top_p)

async def quick_chat_stream(prompt: str, 
                           temperature: float = 0.7,
                           top_p: float = 0.8) -> AsyncGenerator[str, None]:
    """快速聊天（流式）"""
    client = CerebrasClient()
    async for chunk in client.chat_stream(prompt, temperature, top_p):
        yield chunk

# 监控函数
async def monitor_keys(interval: int = 10):
    """监控密钥状态"""
    client = CerebrasClient()
    while True:
        status = client.get_status()
        app_logger.info(f"\n=== 密钥状态监控 ({datetime.now().strftime('%H:%M:%S')}) ===")
        app_logger.info(f"总密钥数: {status['total_keys']}")
        app_logger.info(f"可用密钥: {status['available_keys']}")
        app_logger.info(f"失败密钥: {status['failed_keys']}")
        
        if status['recovery_info']:
            app_logger.info("\n失败密钥恢复信息:")
            for key, info in status['recovery_info'].items():
                app_logger.info(f"  {key}: 失败于 {info['failed_at']}，将在 {info['recovery_in']} 后恢复")
        
        await asyncio.sleep(interval)

# 示例使用
async def main():
    client = CerebrasClient()
    
    # 非流式示例
    app_logger.info("=== 非流式聊天 ===")
    try:
        response = await client.chat("你好，请介绍一下自己，你是谁？")
        app_logger.info(response)
    except Exception as e:
        app_logger.info(f"错误: {e}")
    
    # 流式示例
    app_logger.info("\n=== 流式聊天 ===")
    try:
        async for chunk in client.chat_stream("写一首关于春天的短诗"):
            app_logger.info(chunk, end="", flush=True)
        app_logger.info("")  # 换行
    except Exception as e:
        app_logger.info(f"错误: {e}")
    
    # 查看状态
    app_logger.info("\n=== 客户端状态 ===")
    status = client.get_status()
    app_logger.info(f"总密钥数: {status['total_keys']}")
    app_logger.info(f"可用密钥: {status['available_keys']}")
    app_logger.info(f"失败密钥: {status['failed_keys']}")

# 高级示例：密钥恢复测试
async def recovery_test():
    """测试密钥恢复机制"""
    client = CerebrasClient()
    
    app_logger.info("=== 密钥恢复测试 ===")
    
    # 创建一个监控任务
    monitor_task = asyncio.create_task(monitor_keys(5))
    
    try:
        # 模拟多次调用导致密钥失败
        for i in range(5):
            try:
                app_logger.info(f"\n尝试调用 {i+1}:")
                response = await client.chat("测试消息", max_retries=1)
                app_logger.info(f"成功: {response[:50]}...")
            except Exception as e:
                app_logger.info(f"失败: {e}")
            
            await asyncio.sleep(2)
        
        # 等待一段时间观察恢复
        app_logger.info("\n等待密钥恢复...")
        await asyncio.sleep(70)
        
        # 再次尝试
        app_logger.info("\n恢复后再次尝试:")
        response = await client.chat("测试恢复后的调用")
        app_logger.info(f"成功: {response[:50]}...")
        
    finally:
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

# 错误处理示例
async def error_handling_example():
    client = CerebrasClient()
    
    app_logger.info("=== 错误处理示例 ===")
    
    # 测试重试机制
    for i in range(3):
        try:
            app_logger.info(f"\n尝试 {i+1}:")
            response = await client.chat("测试消息", max_retries=2)
            app_logger.info(f"成功: {response[:50]}...")
            
            # 显示当前状态
            status = client.get_status()
            app_logger.info(f"当前可用密钥: {status['available_keys']}/{status['total_keys']}")
            
        except Exception as e:
            app_logger.info(f"最终失败: {e}")

if __name__ == "__main__":
    # 运行基础示例
    asyncio.run(main())
    
    # 运行恢复测试（取消注释以测试）
    # asyncio.run(recovery_test())
