#api/index.py
#client_server.py
from quart import Quart, request, jsonify, Response
from quart_cors import cors
import asyncio
import json
import uuid
import time
import tempfile
import os
import random
import threading
from typing import *
from concurrent.futures import ThreadPoolExecutor
import logging
import requests
import base64
from pathlib import Path
import io

# 直接集成所有客户端逻辑
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.logger import get_logger
logger = get_logger("server")
# Qwen 客户端
try:
    from client.qwen_client import quick_chat, quick_stream, cleanup_client as qwen_cleanup
except ImportError as e:
    logger.info(f"导入 Qwen 客户端失败: {e}")
    quick_chat = quick_stream = qwen_cleanup = None

# Chutes 客户端
try:
    from client.chutes_client import quick_chat as chutes_chat, quick_chat_stream as chutes_stream
except ImportError as e:
    logger.info(f"导入 Chutes 客户端失败: {e}")
    chutes_chat = chutes_stream = None

# Minimax 客户端
try:
    from client.minimax_client import chat_non_stream as minimax_chat, chat_stream as minimax_stream
except ImportError as e:
    logger.info(f"导入 Minimax 客户端失败: {e}")
    minimax_chat = minimax_stream = None

# Ollama 客户端
try:
    from client.ollama_client import OllamaClient
except ImportError as e:
    logger.info(f"导入 Ollama 客户端失败: {e}")
    OllamaClient = None

# Suanli 客户端
try:
    from client.suanli_client import SuanliClient
except ImportError as e:
    logger.info(f"导入 Suanli 客户端失败: {e}")
    SuanliClient = None

# TTS 客户端
try:
    from client.tts_client import tts
except ImportError as e:
    logger.info(f"导入 TTS 客户端失败: {e}")
    tts = None

# Embed 客户端
try:
    from client.embed_client import EmbedClient
except ImportError as e:
    logger.info(f"导入 Embed 客户端失败: {e}")
    EmbedClient = None

# OpenRouter 客户端
try:
    from client.openrouter_client import quick_chat as openrouter_chat, quick_chat_stream as openrouter_stream
except ImportError as e:
    logger.info(f"导入 OpenRouter 客户端失败: {e}")
    openrouter_chat = openrouter_stream = None

# Cerebras 客户端
try:
    from client.cerebras_client import quick_chat as cerebras_chat, quick_chat_stream as cerebras_stream
except ImportError as e:
    logger.info(f"导入 Cerebras 客户端失败: {e}")
    cerebras_chat = cerebras_stream = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Quart(__name__)
app = cors(app)

class ModelConfig:
    """模型配置类 - 定义各种AI模型的能力和限制"""
    
    QWEN = {
        "name": "QWEN",
        "context_length": 40960,
        "supports_multimodal": True,
        "supports_vision": True,
        "supports_audio": True,
        "supports_document": True,
        "supports_video": True,
        "max_doc_chars": 131072,
        "max_files": 20,  # 支持多文件
        "can_generate_images": True
    }
    
    OPENROUTER = {
        "name": "OPENROUTER", 
        "context_length": 2000000,
        "supports_multimodal": False,
        "supports_vision": True,
        "supports_audio": False,
        "supports_document": False,
        "supports_video": False,
        "max_files": 1,
        "can_generate_images": False
    }
    
    CEREBRAS = {
        "name": "CEREBRAS",
        "context_length": 65536,  # 基于 max_completion_tokens
        "supports_multimodal": False,
        "supports_vision": False,
        "supports_audio": False,
        "supports_document": False,
        "supports_video": False,
        "max_files": 0,
        "can_generate_images": False
    }
    
    CHUTES = {
        "name": "CHUTES", 
        "context_length": 10000,
        "supports_multimodal": False,
        "supports_vision": False,
        "supports_audio": False,
        "supports_document": False,
        "supports_video": False,
        "max_files": 0,
        "can_generate_images": False
    }
    
    MINIMAX = {
        "name": "MINIMAX",
        "context_length": 500000,
        "supports_multimodal": True,
        "supports_vision": True,
        "supports_audio": False,
        "supports_document": False,
        "supports_video": False,
        "max_files": 1,  # 只支持单文件
        "can_generate_images": False
    }
    
    OLLAMA = {
        "name": "OLLAMA",
        "context_length": 128000,
        "supports_multimodal": False,
        "supports_vision": False,
        "supports_audio": False,
        "supports_document": False,
        "supports_video": False,
        "max_files": 0,
        "can_generate_images": False
    }
    
    SUANLI = {
        "name": "SUANLI",
        "context_length": 20480,
        "supports_multimodal": False,
        "supports_vision": False,
        "supports_audio": False,
        "supports_document": False,
        "supports_video": False,
        "max_files": 0,
        "can_generate_images": False
    }
    
    # 回退顺序
    FALLBACK_ORDER = [QWEN, OPENROUTER, CEREBRAS, CHUTES, MINIMAX, OLLAMA, SUANLI]

class FileProcessor:
    """文件处理器 - 处理各种文件类型的识别和转换"""
    
    @staticmethod
    def is_url(path: str) -> bool:
        """判断路径是否为URL地址"""
        return path.startswith(('http://', 'https://'))
    
    @staticmethod
    def is_image(path: str) -> bool:
        """判断文件是否为图片类型"""
        image_exts = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg', '.tiff', '.ico')
        return path.lower().endswith(image_exts) or ('image' in path.lower())
    
    @staticmethod
    def is_video(path: str) -> bool:
        """判断文件是否为视频类型"""
        video_exts = ('.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv', '.3gp', '.m4v')
        return path.lower().endswith(video_exts) or ('video' in path.lower())
    
    @staticmethod
    def is_audio(path: str) -> bool:
        """判断文件是否为音频类型"""
        audio_exts = ('.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a', '.opus')
        return path.lower().endswith(audio_exts) or ('audio' in path.lower())
    
    @staticmethod
    def is_document(path: str) -> bool:
        """判断文件是否为文档类型"""
        doc_exts = ('.pdf', '.doc', '.docx', '.txt', '.md', '.rtf', '.csv', '.xlsx', '.ppt', '.pptx')
        return path.lower().endswith(doc_exts)
    
    @staticmethod
    async def download_file(url: str, temp_dir: str) -> str:
        """下载网络文件到本地临时文件"""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: requests.get(url, timeout=30))
            if response.status_code == 200:
                content = response.content
                # 从URL或Content-Type推断文件扩展名
                content_type = response.headers.get('content-type', '')
                if 'image' in content_type:
                    ext = content_type.split('/')[-1] if '/' in content_type else 'jpg'
                elif 'video' in content_type:
                    ext = content_type.split('/')[-1] if '/' in content_type else 'mp4'
                elif 'audio' in content_type:
                    ext = content_type.split('/')[-1] if '/' in content_type else 'mp3'
                elif 'pdf' in content_type:
                    ext = 'pdf'
                else:
                    # 从URL推断
                    ext = url.split('.')[-1] if '.' in url else 'bin'
                
                temp_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}.{ext}")
                with open(temp_path, 'wb') as f:
                    f.write(content)
                return temp_path
            else:
                raise Exception(f"下载失败: HTTP {response.status_code}")
        except Exception as e:
            raise Exception(f"下载文件失败: {str(e)}")
    
    @staticmethod
    def text_to_temp_document(text: str, max_chars: int = 131072) -> str:
        """将文本转换为临时文档文件"""
        if len(text) > max_chars:
            text = text[:max_chars]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(text)
            return f.name
    
    @staticmethod
    def analyze_files(file_paths: Union[str, List[str]]) -> Dict[str, List[str]]:
        """分析文件类型并分类"""
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        
        result = {
            'images': [],
            'videos': [],
            'audios': [],
            'documents': [],
            'unknown': []
        }
        
        for path in file_paths:
            if FileProcessor.is_image(path):
                result['images'].append(path)
            elif FileProcessor.is_video(path):
                result['videos'].append(path)
            elif FileProcessor.is_audio(path):
                result['audios'].append(path)
            elif FileProcessor.is_document(path):
                result['documents'].append(path)
            else:
                result['unknown'].append(path)
        
        return result

class ClientHandler:
    """统一客户端处理器 - 优化高并发性能，支持多模型调用和回退机制"""
    
    def __init__(self, max_concurrent: int = 100):
        """
        初始化客户端处理器
        
        Args:
            max_concurrent: 最大并发数
        """
        # 大幅提升并发限制
        self.max_concurrent = max_concurrent
        self.chat_semaphore = asyncio.Semaphore(max_concurrent)
        self.embed_semaphore = asyncio.Semaphore(10)  # 嵌入请求限制
        self.download_semaphore = asyncio.Semaphore(20)  # 下载文件限制
        
        # 使用更大的线程池以支持高并发
        self.thread_pool = ThreadPoolExecutor(max_workers=50)
        
        # 初始化客户端
        self.ollama_client = None
        self.suanli_client = None
        self.embed_client = None
        self._client_lock = asyncio.Lock()
        
        # 临时文件管理
        self.temp_dir = tempfile.mkdtemp()
        self.temp_files = set()
        self._temp_lock = asyncio.Lock()
        
        # 统计信息
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'model_usage': {config['name']: 0 for config in ModelConfig.FALLBACK_ORDER},
            'fallback_usage': 0,
            'file_downloads': 0,
            'multimodal_rejections': 0,
            'direct_model_calls': 0
        }
    
    async def _get_ollama_client(self) -> OllamaClient:
        """获取Ollama客户端实例（单例模式）"""
        if self.ollama_client is None:
            async with self._client_lock:
                if self.ollama_client is None:
                    self.ollama_client = OllamaClient()
        return self.ollama_client
    
    async def _get_suanli_client(self) -> SuanliClient:
        """获取Suanli客户端实例（单例模式）"""
        if self.suanli_client is None:
            async with self._client_lock:
                if self.suanli_client is None:
                    self.suanli_client = SuanliClient()
        return self.suanli_client
    
    async def _get_embed_client(self) -> EmbedClient:
        """获取嵌入客户端实例（单例模式）"""
        if self.embed_client is None:
            async with self._client_lock:
                if self.embed_client is None:
                    self.embed_client = EmbedClient()
        return self.embed_client
    
    async def _add_temp_file(self, file_path: str) -> None:
        """添加临时文件到管理列表"""
        async with self._temp_lock:
            self.temp_files.add(file_path)
    
    async def _remove_temp_file(self, file_path: str) -> None:
        """从管理列表移除临时文件"""
        async with self._temp_lock:
            self.temp_files.discard(file_path)
            if os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                except:
                    pass
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """截断文本到指定长度"""
        if len(text) <= max_length:
            return text
        return text[:max_length]
    
    def _validate_files_for_model(self, model_config: Dict, file_paths: List[str]) -> Tuple[bool, str]:
        """
        验证模型是否能处理指定的文件
        
        Args:
            model_config: 模型配置字典
            file_paths: 文件路径列表
            
        Returns:
            (是否可以处理, 错误消息)
        """
        if not file_paths:
            return True, ""
        
        # 检查文件数量限制
        if len(file_paths) > model_config.get('max_files', 0):
            if model_config['name'] == 'QWEN':
                return False, f"文件数量超限: {len(file_paths)} > {model_config['max_files']}"
            else:
                return False, f"模型 {model_config['name']} 不支持多文件，当前有 {len(file_paths)} 个文件"
        
        # 分析文件类型
        file_analysis = FileProcessor.analyze_files(file_paths)
        
        # 检查各种文件类型支持
        if file_analysis['images'] and not model_config.get('supports_vision'):
            return False, f"模型 {model_config['name']} 不支持图像文件"
        
        if file_analysis['videos'] and not model_config.get('supports_video'):
            return False, f"模型 {model_config['name']} 不支持视频文件"
        
        if file_analysis['audios'] and not model_config.get('supports_audio'):
            return False, f"模型 {model_config['name']} 不支持音频文件"
        
        if file_analysis['documents'] and not model_config.get('supports_document'):
            return False, f"模型 {model_config['name']} 不支持文档文件"
        
        if file_analysis['unknown']:
            return False, f"模型 {model_config['name']} 不支持未知类型文件: {file_analysis['unknown']}"
        
        return True, ""
    
    def _should_handle_as_document(self, text: str, model_config: Dict) -> Tuple[bool, Optional[str]]:
        """
        判断是否应该将长文本转换为文档处理
        
        Args:
            text: 输入文本
            model_config: 模型配置
            
        Returns:
            (是否需要转换为文档, 文档路径)
        """
        if (len(text) > model_config['context_length'] and 
            model_config.get('supports_document') and 
            len(text) <= model_config.get('max_doc_chars', 0)):
            
            doc_path = FileProcessor.text_to_temp_document(text, model_config.get('max_doc_chars'))
            asyncio.create_task(self._add_temp_file(doc_path))
            return True, doc_path
        return False, None
    
    async def _try_qwen(self, text: str, file_paths: Optional[List[str]] = None, stream: bool = False) -> Union[str, AsyncGenerator]:
        """
        尝试使用QWEN模型
        
        Args:
            text: 输入文本
            file_paths: 文件路径列表
            stream: 是否流式输出
            
        Returns:
            模型响应或异步生成器
        """
        config = ModelConfig.QWEN
        # 检查是否需要转换为文档
        should_doc, doc_path = self._should_handle_as_document(text, config)
        if should_doc:
            file_paths = [doc_path] if not file_paths else (file_paths + [doc_path])
        else:
            text = self._truncate_text(text, config['context_length'])
        
        try:
            if stream:
                # 直接返回原始生成器，不使用超时包装
                return quick_stream(text, file_paths)
            else:
                result = await quick_chat(text, file_paths)
                return result
        except Exception as e:
            raise e
    
    async def _try_openrouter(self, text: str, file_paths: Optional[List[str]] = None, stream: bool = False) -> Union[str, AsyncGenerator]:
        """
        尝试使用OPENROUTER模型
        
        Args:
            text: 输入文本
            file_paths: 文件路径列表（仅支持单个图片）
            stream: 是否流式输出
            
        Returns:
            模型响应或异步生成器
        """
        config = ModelConfig.OPENROUTER
        text = self._truncate_text(text, config['context_length'])
        
        # OPENROUTER只支持单个图片
        image_path = ""
        if file_paths:
            file_analysis = FileProcessor.analyze_files(file_paths)
            if file_analysis['images']:
                image_path = file_analysis['images'][0]  # 只取第一个图片
        
        try:
            if stream:
                return openrouter_stream(text, image_path)
            else:
                result = await openrouter_chat(text, image_path)
                return result
        except Exception as e:
            raise e
    
    async def _try_cerebras(self, text: str, stream: bool = False) -> Union[str, AsyncGenerator]:
        """
        尝试使用CEREBRAS模型（仅支持文本）
        
        Args:
            text: 输入文本
            stream: 是否流式输出
            
        Returns:
            模型响应或异步生成器
        """
        config = ModelConfig.CEREBRAS
        text = self._truncate_text(text, config['context_length'])
        
        if stream:
            return cerebras_stream(text)
        else:
            return await cerebras_chat(text)
    
    async def _try_chutes(self, text: str, stream: bool = False) -> Union[str, AsyncGenerator]:
        """
        尝试使用CHUTES模型（仅支持文本）
        
        Args:
            text: 输入文本
            stream: 是否流式输出
            
        Returns:
            模型响应或异步生成器
        """
        config = ModelConfig.CHUTES
        text = self._truncate_text(text, config['context_length'])
        
        if stream:
            return chutes_stream(text)
        else:
            return await chutes_chat(text)
    
    async def _try_minimax(self, text: str, file_paths: Optional[List[str]] = None, stream: bool = False) -> Union[str, AsyncGenerator]:
        """
        尝试使用MINIMAX模型
        
        Args:
            text: 输入文本
            file_paths: 文件路径列表（仅支持单个图片）
            stream: 是否流式输出
            
        Returns:
            模型响应或异步生成器
        """
        config = ModelConfig.MINIMAX
        text = self._truncate_text(text, config['context_length'])
        
        # MINIMAX只支持单个图片
        image_path = None
        if file_paths:
            file_analysis = FileProcessor.analyze_files(file_paths)
            if file_analysis['images']:
                image_path = file_analysis['images'][0]  # 只取第一个图片
        
        if stream:
            return minimax_stream(text, image_path)
        else:
            return await minimax_chat(text, image_path)
    
    async def _try_ollama(self, text: str, stream: bool = False) -> Union[str, AsyncGenerator]:
        """
        尝试使用OLLAMA模型（仅支持文本）
        
        Args:
            text: 输入文本
            stream: 是否流式输出
            
        Returns:
            模型响应或异步生成器
        """
        config = ModelConfig.OLLAMA
        text = self._truncate_text(text, config['context_length'])
        
        client = await self._get_ollama_client()
        
        if stream:
            return client.chat_stream(text)
        else:
            return await client.chat(text)
    
    async def _try_suanli(self, text: str, stream: bool = False) -> Union[str, AsyncGenerator]:
        """
        尝试使用SUANLI模型（仅支持文本）
        
        Args:
            text: 输入文本
            stream: 是否流式输出
            
        Returns:
            模型响应或异步生成器
        """
        config = ModelConfig.SUANLI
        text = self._truncate_text(text, config['context_length'])
        
        client = await self._get_suanli_client()
        
        if stream:
            async def suanli_stream_generator():
                loop = asyncio.get_event_loop()
                for chunk in await loop.run_in_executor(None, lambda: client.chat_stream_generator(text, show_stats=False)):
                    if not chunk.startswith("❌"):
                        yield chunk
            
            return suanli_stream_generator()
        else:
            loop = asyncio.get_event_loop()
            result, _ = await loop.run_in_executor(None, lambda: client.chat(text, show_stats=False))
            if result is None:
                raise Exception("Suanli返回None")
            return result
    
    async def _execute_with_fallback(self, text: str, file_paths: Optional[List[str]] = None, stream: bool = False, retries: int = 0) -> Union[str, AsyncGenerator]:
        """
        执行带回退逻辑的模型调用
        
        Args:
            text: 输入文本
            file_paths: 文件路径列表
            stream: 是否流式输出
            retries: 重试次数
            
        Returns:
            模型响应或异步生成器
            
        Raises:
            Exception: 所有模型调用失败
        """
        async with self.chat_semaphore:
            self.stats['total_requests'] += 1
            
            # 处理文件URL
            processed_files = []
            if file_paths:
                try:
                    processed_files = file_paths
                except Exception as e:
                    raise Exception(f"文件处理失败: {str(e)}")
            
            for attempt in range(retries + 1):
                last_error = None
                
                for config in ModelConfig.FALLBACK_ORDER:
                    # 验证模型是否能处理文件
                    can_handle, error_msg = self._validate_files_for_model(config, processed_files)
                    
                    if not can_handle:
                        if config['name'] == 'QWEN':
                            # 如果连QWEN都不能处理，直接返回错误
                            self.stats['multimodal_rejections'] += 1
                            raise Exception(error_msg)
                        continue
                    try:
                        if config['name'] == 'QWEN':
                            result = await self._try_qwen(text, processed_files, stream)
                        elif config['name'] == 'OPENROUTER':
                            result = await self._try_openrouter(text, processed_files, stream)
                        elif config['name'] == 'CEREBRAS':
                            result = await self._try_cerebras(text, stream)
                        elif config['name'] == 'CHUTES':
                            result = await self._try_chutes(text, stream)
                        elif config['name'] == 'MINIMAX':
                            result = await self._try_minimax(text, processed_files, stream)
                        elif config['name'] == 'OLLAMA':
                            result = await self._try_ollama(text, stream)
                        elif config['name'] == 'SUANLI':
                            result = await self._try_suanli(text, stream)
                        
                        self.stats['successful_requests'] += 1
                        self.stats['model_usage'][config['name']] += 1
                        return result
                        
                    except Exception as e:
                        last_error = e
                        logger.warning(f"模型 {config['name']} 调用失败 (尝试 {attempt + 1}): {str(e)}")
                        continue
                
                # 如果不是最后一次尝试，等待一段时间再重试
                if attempt < retries:
                    await asyncio.sleep(1)
            
            # 所有模型和重试都失败了
            self.stats['fallback_usage'] += 1
            error_message = f"所有模型调用失败，最后一个错误: {str(last_error)}" if last_error else "所有模型调用失败"
            raise Exception(error_message)
    
    async def chat_with_specific_model(self, model_name: str, text: str, file_paths: Optional[List[str]] = None, stream: bool = False, retries: int = 0) -> Union[str, AsyncGenerator]:
        """
        调用指定的模型（不使用回退机制）
        
        Args:
            model_name: 模型名称
            text: 输入文本
            file_paths: 文件路径列表
            stream: 是否流式输出
            retries: 重试次数
            
        Returns:
            模型响应或异步生成器
            
        Raises:
            Exception: 模型调用失败或不支持的模型
        """
        async with self.chat_semaphore:
            self.stats['total_requests'] += 1
            self.stats['direct_model_calls'] += 1
            
            model_name = model_name.upper()
            
            # 验证模型是否存在
            model_configs = {config['name']: config for config in ModelConfig.FALLBACK_ORDER}
            if model_name not in model_configs:
                raise Exception(f"不支持的模型: {model_name}")
            
            config = model_configs[model_name]
            
            # 验证文件是否能被模型处理
            can_handle, error_msg = self._validate_files_for_model(config, file_paths or [])
            if not can_handle:
                raise Exception(error_msg)
            
            # 处理文件
            processed_files = file_paths or []
            
            for attempt in range(retries + 1):
                try:
                    if model_name == 'QWEN':
                        result = await self._try_qwen(text, processed_files, stream)
                    elif model_name == 'OPENROUTER':
                        result = await self._try_openrouter(text, processed_files, stream)
                    elif model_name == 'CEREBRAS':
                        result = await self._try_cerebras(text, stream)
                    elif model_name == 'CHUTES':
                        result = await self._try_chutes(text, stream)
                    elif model_name == 'MINIMAX':
                        result = await self._try_minimax(text, processed_files, stream)
                    elif model_name == 'OLLAMA':
                        result = await self._try_ollama(text, stream)
                    elif model_name == 'SUANLI':
                        result = await self._try_suanli(text, stream)
                    else:
                        raise Exception(f"未实现的模型调用: {model_name}")
                    
                    self.stats['successful_requests'] += 1
                    self.stats['model_usage'][model_name] += 1
                    return result
                    
                except Exception as e:
                    if attempt < retries:
                        await asyncio.sleep(1)
                        continue
                    raise e
    
    async def chat_with_model_stream(self, text: str, file_paths: Optional[List[str]] = None, retries: int = 0) -> AsyncGenerator[str, None]:
        """
        流式聊天接口（使用回退机制）
        
        Args:
            text: 输入文本
            file_paths: 文件路径列表
            retries: 重试次数
            
        Yields:
            模型响应的文本块
        """
        result = await self._execute_with_fallback(text, file_paths, stream=True, retries=retries)
        async for chunk in result:
            yield chunk
    
    async def chat_with_model(self, text: str, file_paths: Optional[List[str]] = None, retries: int = 0) -> str:
        """
        非流式聊天接口（使用回退机制）
        
        Args:
            text: 输入文本
            file_paths: 文件路径列表
            retries: 重试次数
            
        Returns:
            模型响应文本
        """
        result = await self._execute_with_fallback(text, file_paths, stream=False, retries=retries)
        return result
    
    async def chat_with_tts(self, text: str, voice: str = "派蒙", save_path: Optional[str] = None, retries: int = 0) -> Dict[str, Any]:
        """
        文本转语音接口
        
        Args:
            text: 输入文本
            voice: 语音类型
            save_path: 保存路径
            retries: 重试次数
            
        Returns:
            TTS结果字典
        """
        for attempt in range(retries + 1):
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: tts(text, voice, save_path))
                return result
            except Exception as e:
                if attempt < retries:
                    await asyncio.sleep(1)
                    continue
                raise e
    
    async def chat_with_embed(self, text: str, retries: int = 0) -> List[float]:
        """
        文本嵌入接口
        
        Args:
            text: 输入文本
            retries: 重试次数
            
        Returns:
            嵌入向量
        """
        logger.info(text)
        async with self.embed_semaphore:
            for attempt in range(retries + 1):
                try:
                    client = await self._get_embed_client()
                    loop = asyncio.get_event_loop()
                    embedding = await loop.run_in_executor(None, lambda: client.get_embedding(text))
                    return embedding
                except Exception as e:
                    if attempt < retries:
                        await asyncio.sleep(1)
                        continue
                    raise e
    
    async def cleanup_temp_files(self, file_paths: List[str]) -> None:
        """清理临时文件"""
        for file_path in file_paths:
            await self._remove_temp_file(file_path)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            包含各种统计数据的字典
        """
        success_rate = (self.stats['successful_requests'] / max(self.stats['total_requests'], 1)) * 100
        return {
            **self.stats,
            'success_rate': round(success_rate, 2),
            'fallback_rate': round((self.stats['fallback_usage'] / max(self.stats['total_requests'], 1)) * 100, 2),
            'direct_call_rate': round((self.stats['direct_model_calls'] / max(self.stats['total_requests'], 1)) * 100, 2)
        }
    
    async def close(self) -> None:
        """清理资源"""
        self.thread_pool.shutdown(wait=True)
        
        # 清理所有临时文件
        async with self._temp_lock:
            for file_path in list(self.temp_files):
                await self._remove_temp_file(file_path)
            
            # 删除临时目录
            try:
                os.rmdir(self.temp_dir)
            except:
                pass
        
        try:
            await qwen_cleanup()
        except:
            pass

# 全局处理器实例
_global_handler = None
_handler_lock = asyncio.Lock()

async def get_handler() -> ClientHandler:
    """获取全局处理器实例（单例模式）"""
    global _global_handler
    if _global_handler is None:
        async with _handler_lock:
            if _global_handler is None:
                _global_handler = ClientHandler()
    return _global_handler

def create_error_response(error_type: str, message: str, code: int = 500) -> Dict[str, Any]:
    """创建标准错误响应"""
    return {
        "error": {
            "type": error_type,
            "message": message,
            "code": code
        }
    }

def create_openai_response(content: str, model: str, usage: dict = None) -> Dict[str, Any]:
    """创建OpenAI格式的响应"""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": "stop"
        }],
        "usage": usage or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }

def create_anthropic_response(content: str, model: str, usage: dict = None) -> Dict[str, Any]:
    """创建Anthropic格式的响应"""
    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": content
            }
        ],
        "model": model,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": usage or {
            "input_tokens": 0,
            "output_tokens": 0
        }
    }

async def create_openai_stream_response(content_generator: AsyncGenerator, model: str):
    """创建OpenAI格式的流式响应"""
    # 首先发送开始标记
    yield f"data: {json.dumps({'id': f'chatcmpl-{uuid.uuid4().hex}', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
    
    # 发送内容
    async for chunk in content_generator:
        if chunk:
            yield f"data: {json.dumps({'id': f'chatcmpl-{uuid.uuid4().hex}', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model, 'choices': [{'index': 0, 'delta': {'content': chunk}, 'finish_reason': None}]})}\n\n"
    
    # 发送结束标记
    yield f"data: {json.dumps({'id': f'chatcmpl-{uuid.uuid4().hex}', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
    yield "data: [DONE]\n\n"

async def create_anthropic_stream_response(content_generator: AsyncGenerator, model: str):
    """创建Anthropic格式的流式响应"""
    message_id = f"msg_{uuid.uuid4().hex}"
    
    # 发送开始事件
    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': message_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': model, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
    
    # 发送内容开始事件
    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
    
    # 发送内容增量
    async for chunk in content_generator:
        if chunk:
            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': chunk}})}\n\n"
    
    # 发送结束事件
    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n"
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

def extract_text_from_messages(messages: List[Dict], format_type: str = "openai") -> str:
    """从消息列表中提取文本内容"""
    text_content = ""
    
    if format_type == "openai":
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            if isinstance(content, str):
                text_content += f"{role}: {content}\n"
            elif isinstance(content, list):
                # 处理多模态内容
                for item in content:
                    if item.get('type') == 'text':
                        text_content += f"{role}: {item.get('text', '')}\n"
    
    elif format_type == "anthropic":
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            if isinstance(content, str):
                text_content += f"{role}: {content}\n"
            elif isinstance(content, list):
                # 处理多模态内容
                for item in content:
                    if item.get('type') == 'text':
                        text_content += f"{role}: {item.get('text', '')}\n"
    
    return text_content

def extract_files_from_messages(messages: List[Dict], format_type: str = "openai") -> List[str]:
    """从消息列表中提取文件URL"""
    file_urls = []
    
    if format_type == "openai":
        for message in messages:
            content = message.get('content', [])
            if isinstance(content, list):
                for item in content:
                    # 支持file_url格式 (新增)
                    if item.get('type') == 'file_url':
                        file_url = item.get('file_url', {}).get('url')
                        if file_url:
                            file_urls.append(file_url)
                    # 兼容原有的image_url格式
                    elif item.get('type') == 'image_url':
                        image_url = item.get('image_url', {}).get('url')
                        if image_url:
                            file_urls.append(image_url)
    
    elif format_type == "anthropic":
        for message in messages:
            content = message.get('content', [])
            if isinstance(content, list):
                for item in content:
                    item_type = item.get('type')
                    
                    # 处理各种文件类型
                    if item_type in ['image', 'video', 'audio', 'document']:
                        source = item.get('source', {})
                        if source.get('type') == 'url':
                            # 直接URL
                            url = source.get('url')
                            if url:
                                file_urls.append(url)
                        elif source.get('type') == 'base64':
                            # 处理base64数据，转换为临时文件
                            media_type = source.get('media_type', '')
                            data = source.get('data', '')
                            if data:
                                # 根据media_type确定文件扩展名
                                if 'image' in media_type:
                                    ext = media_type.split('/')[-1] if '/' in media_type else 'jpg'
                                elif 'video' in media_type:
                                    ext = media_type.split('/')[-1] if '/' in media_type else 'mp4'
                                elif 'audio' in media_type:
                                    ext = media_type.split('/')[-1] if '/' in media_type else 'mp3'
                                elif 'pdf' in media_type:
                                    ext = 'pdf'
                                else:
                                    ext = 'bin'
                                
                                temp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.{ext}")
                                try:
                                    with open(temp_path, 'wb') as f:
                                        f.write(base64.b64decode(data))
                                    file_urls.append(temp_path)
                                except Exception as e:
                                    logger.warning(f"处理base64文件失败: {str(e)}")
    
    return file_urls

@app.route('/v1/chat/completions', methods=['POST'])
async def chat_completions():
    """OpenAI兼容的聊天完成API - 支持多模型调用和文件处理"""
    temp_files = []
    try:
        data = await request.get_json()
        # 验证必需参数
        if not data or 'messages' not in data:
            return jsonify(create_error_response("invalid_request", "缺少messages参数", 400)), 400
        
        model = data.get('model', 'auto_chat')
        messages = data.get('messages', [])
        stream = data.get('stream', False)
        
        # 处理options参数
        options = data.get('options', {})
        retries = options.get('retries', 0)
        
        # 组合消息内容
        text_content = extract_text_from_messages(messages, "openai")
        
        if not text_content.strip():
            return jsonify(create_error_response("invalid_request", "消息内容为空", 400)), 400
        
        handler = await get_handler()
        
        # 处理文件上传
        file_urls = []
        files = await request.files
        if files:
            for file_key in files:
                file = files[file_key]
                if file.filename:
                    # 保存上传的文件
                    file_path = os.path.join(handler.temp_dir, f"{uuid.uuid4().hex}_{file.filename}")
                    await file.save(file_path)
                    await handler._add_temp_file(file_path)
                    file_urls.append(file_path)
                    temp_files.append(file_path)
        
        # 从消息中提取文件URL
        message_files = extract_files_from_messages(messages, "openai")
        file_urls.extend(message_files)
        temp_files.extend([f for f in message_files if not f.startswith(('http://', 'https://'))])
        
        # 获取模型配置
        model_configs = {config['name']: config for config in ModelConfig.FALLBACK_ORDER}
        model_upper = model.upper()
        
        try:
            if model_upper in model_configs:
                # 直接调用指定模型
                if stream:
                    async def generate():
                        try:
                            async for chunk in create_openai_stream_response(
                                await handler.chat_with_specific_model(model_upper, text_content, file_urls or None, True, retries), 
                                model
                            ):
                                yield chunk
                        except Exception as e:
                            error_chunk = f"data: {json.dumps(create_error_response('api_error', str(e)))}\n\n"
                            yield error_chunk
                            yield "data: [DONE]\n\n"
                        finally:
                            # 清理临时文件
                            await handler.cleanup_temp_files(temp_files)
                    
                    return Response(generate(), mimetype='text/plain')
                else:
                    content = await handler.chat_with_specific_model(model_upper, text_content, file_urls or None, False, retries)
                    return jsonify(create_openai_response(content, model))
            
            elif model in ['auto_chat', 'gpt-4', 'gpt-4.1'] or model.startswith("claude"):
                # 使用回退机制
                if stream:
                    async def generate():
                        try:
                            async for chunk in create_openai_stream_response(
                                handler.chat_with_model_stream(text_content, file_urls or None, retries), 
                                model
                            ):
                                yield chunk
                        except Exception as e:
                            error_chunk = f"data: {json.dumps(create_error_response('api_error', str(e)))}\n\n"
                            yield error_chunk
                            yield "data: [DONE]\n\n"
                        finally:
                            # 清理临时文件
                            await handler.cleanup_temp_files(temp_files)
                    
                    return Response(generate(), mimetype='text/plain')
                else:
                    content = await handler.chat_with_model(text_content, file_urls or None, retries)
                    return jsonify(create_openai_response(content, model))
            
            elif model == 'auto_tts':
                try:
                    voice = data.get('voice', '派蒙')
                    result = await handler.chat_with_tts(text_content, voice, retries=retries)
                    return jsonify(result)
                except Exception as e:
                    return jsonify(create_error_response("tts_error", str(e), 500)), 500
            
            elif model == 'auto_embedding':
                try:
                    embedding = await handler.chat_with_embed(text_content, retries=retries)
                    return jsonify({
                        "object": "list",
                        "data": [{
                            "object": "embedding",
                            "embedding": embedding,
                            "index": 0
                        }],
                        "model": model,
                        "usage": {
                            "prompt_tokens": len(text_content.split()),
                            "total_tokens": len(text_content.split())
                        }
                    })
                except Exception as e:
                    return jsonify(create_error_response("embedding_error", str(e), 500)), 500
            
            else:
                return jsonify(create_error_response("invalid_model", f"不支持的模型: {model}", 400)), 400
        
        except Exception as e:
            return jsonify(create_error_response("api_error", str(e), 500)), 500
        finally:
            # 清理临时文件
            await handler.cleanup_temp_files(temp_files)
    
    except Exception as e:
        logger.error(f"OpenAI API调用出错: {str(e)}")
        return jsonify(create_error_response("internal_error", str(e), 500)), 500

@app.route('/v1/messages', methods=['POST'])
async def anthropic_messages():
    """Anthropic兼容的消息API - 支持多模型调用和文件处理"""
    temp_files = []
    try:
        data = await request.get_json()

        # 验证必需参数
        if not data or 'messages' not in data:
            return jsonify(create_error_response("invalid_request", "缺少messages参数", 400)), 400
        
        model = data.get('model', 'auto_chat')
        messages = data.get('messages', [])
        stream = data.get('stream', False)
        max_tokens = data.get('max_tokens', 1048576)

        # 处理options参数
        options = data.get('options', {})
        retries = options.get('retries', 0)
        
        # 组合消息内容
        text_content = extract_text_from_messages(messages, "anthropic")
        
        if not text_content.strip():
            return jsonify(create_error_response("invalid_request", "消息内容为空", 400)), 400
        
        handler = await get_handler()
        
        # 处理文件上传
        file_urls = []
        files = await request.files
        if files:
            for file_key in files:
                file = files[file_key]
                if file.filename:
                    # 保存上传的文件
                    file_path = os.path.join(handler.temp_dir, f"{uuid.uuid4().hex}_{file.filename}")
                    await file.save(file_path)
                    await handler._add_temp_file(file_path)
                    file_urls.append(file_path)
                    temp_files.append(file_path)
        
        # 从消息中提取文件
        message_files = extract_files_from_messages(messages, "anthropic")
        file_urls.extend(message_files)
        temp_files.extend([f for f in message_files if not f.startswith(('http://', 'https://'))])
        
        # 获取模型配置
        model_configs = {config['name']: config for config in ModelConfig.FALLBACK_ORDER}
        model_upper = model.upper()
        
        try:
            if model_upper in model_configs:
                # 直接调用指定模型
                if stream:
                    async def generate():
                        try:
                            async for chunk in create_anthropic_stream_response(
                                await handler.chat_with_specific_model(model_upper, text_content, file_urls or None, True, retries), 
                                model
                            ):
                                yield chunk
                        except Exception as e:
                            error_chunk = f"event: error\ndata: {json.dumps(create_error_response('api_error', str(e)))}\n\n"
                            yield error_chunk
                        finally:
                            # 清理临时文件
                            await handler.cleanup_temp_files(temp_files)
                    
                    return Response(generate(), mimetype='text/plain')
                else:
                    content = await handler.chat_with_specific_model(model_upper, text_content, file_urls or None, False, retries)
                    return jsonify(create_anthropic_response(content, model))
            
            elif model.startswith("claude") or model == "auto_chat":
                # 使用回退机制
                if stream:
                    async def generate():
                        try:
                            async for chunk in create_anthropic_stream_response(
                                handler.chat_with_model_stream(text_content, file_urls or None, retries), 
                                model
                            ):
                                yield chunk
                        except Exception as e:
                            error_chunk = f"event: error\ndata: {json.dumps(create_error_response('api_error', str(e)))}\n\n"
                            yield error_chunk
                        finally:
                            # 清理临时文件
                            await handler.cleanup_temp_files(temp_files)
                    
                    return Response(generate(), mimetype='text/plain')
                else:
                    content = await handler.chat_with_model(text_content, file_urls or None, retries)
                    return jsonify(create_anthropic_response(content, model))
            
            else:
                return jsonify(create_error_response("invalid_model", f"不支持的模型: {model}", 400)), 400
        
        except Exception as e:
            return jsonify(create_error_response("api_error", str(e), 500)), 500
        finally:
            # 清理临时文件
            await handler.cleanup_temp_files(temp_files)
    
    except Exception as e:
        logger.error(f"Anthropic API调用出错: {str(e)}")
        return jsonify(create_error_response("internal_error", str(e), 500)), 500

@app.route('/v1/models', methods=['GET'])
async def list_models():
    """列出可用模型 - 兼容OpenAI和Anthropic格式"""
    models = [
        {
            "id": "auto_chat",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "nbot",
            "permission": [],
            "root": "auto_chat",
            "parent": None,
        },
        {
            "id": "QWEN",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "nbot",
            "permission": [],
            "root": "QWEN",
            "parent": None,
        },
        {
            "id": "OPENROUTER",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "nbot",
            "permission": [],
            "root": "OPENROUTER", 
            "parent": None,
        },
        {
            "id": "CEREBRAS",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "nbot",
            "permission": [],
            "root": "CEREBRAS",
            "parent": None,
        },
        {
            "id": "CHUTES",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "nbot",
            "permission": [],
            "root": "CHUTES",
            "parent": None,
        },
        {
            "id": "MINIMAX",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "nbot",
            "permission": [],
            "root": "MINIMAX",
            "parent": None,
        },
        {
            "id": "OLLAMA",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "nbot",
            "permission": [],
            "root": "OLLAMA",
            "parent": None,
        },
        {
            "id": "SUANLI",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "nbot",
            "permission": [],
            "root": "SUANLI",
            "parent": None,
        },
        {
            "id": "claude-3-sonnet-20240229",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "nbot",
            "permission": [],
            "root": "auto_chat",
            "parent": None,
        },
        {
            "id": "claude-3-opus-20240229",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "nbot",
            "permission": [],
            "root": "auto_chat",
            "parent": None,
        },
        {
            "id": "claude-3-haiku-20240307",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "nbot",
            "permission": [],
            "root": "auto_chat",
            "parent": None,
        },
        {
            "id": "gpt-4",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "nbot",
            "permission": [],
            "root": "auto_chat",
            "parent": None,
        },
        {
            "id": "gpt-4.1",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "nbot",
            "permission": [],
            "root": "auto_chat",
            "parent": None,
        },
        {
            "id": "auto_tts",
            "object": "model", 
            "created": int(time.time()),
            "owned_by": "nbot",
            "permission": [],
            "root": "auto_tts",
            "parent": None,
        },
        {
            "id": "auto_embedding",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "nbot", 
            "permission": [],
            "root": "auto_embedding",
            "parent": None,
        }
    ]
    
    return jsonify({
        "object": "list",
        "data": models
    })

@app.route('/v1/health', methods=['GET'])
async def health_check():
    """健康检查和统计信息"""
    handler = await get_handler()
    stats = handler.get_stats()
    return jsonify({
        "status": "healthy",
        "timestamp": int(time.time()),
        "stats": stats
    })

@app.route('/', methods=['GET'])
async def index():
    """根路径 - API文档和功能介绍"""
    return jsonify({
        "message": "OpenAI & Anthropic Compatible API - Multimodal Support with Direct Model Access",
        "version": "2.2.1",
        "endpoints": {
            "openai_chat": "/v1/chat/completions",
            "anthropic_messages": "/v1/messages",
            "models": "/v1/models", 
            "health": "/v1/health"
        },
        "supported_formats": ["OpenAI", "Anthropic"],
        "auto_models": ["auto_chat", "auto_tts", "auto_embedding"],
        "direct_models": ["QWEN", "OPENROUTER", "CEREBRAS", "CHUTES", "MINIMAX", "OLLAMA", "SUANLI"],
        "aliases": {
            # "claude-3-sonnet-20240229": "auto_chat",
            # "claude-3-opus-20240229": "auto_chat", 
            # "claude-3-haiku-20240307": "auto_chat",
            # "gpt-4": "auto_chat",
            # "gpt-4.1": "auto_chat"
            # 这是一个模型映射表，用于适配第三方ide，暂时不用因此注释掉
        },
        "model_capabilities": {
            "QWEN": {
                "multimodal": True,
                "supports": ["image", "video", "audio", "document"],
                "max_files": 20,
                "context_length": 40960,
                "model": "qwen3-coder-plus"
            },
            "OPENROUTER": {
                "multimodal": False,
                "supports": ["image"],
                "max_files": 1,
                "context_length": 2000000,
                "model": "x-ai/grok-4-fast:free"
            },
            "CEREBRAS": {
                "multimodal": False,
                "supports": ["text-only"],
                "max_files": 0,
                "context_length": 65536,
                "model": "qwen-3-coder-480b"
            },
            "CHUTES": {
                "multimodal": False,
                "supports": ["text-only"],
                "max_files": 0,
                "context_length": 10000,
                "model": "zai-org/GLM-4.5-Air"
            },
            "MINIMAX": {
                "multimodal": True,
                "supports": ["image"],
                "max_files": 1,
                "context_length": 500000,
                "model": "minimax"
            },
            "OLLAMA": {
                "multimodal": False,
                "supports": ["text-only"],
                "max_files": 0,
                "context_length": 128000,
                "model": "gpt-oss-120b"
            },
            "SUANLI": {
                "multimodal": False,
                "supports": ["text-only"],
                "max_files": 0,
                "context_length": 20480,
                "model": "free:QwQ-32B"
            }
        },
        "usage_modes": {
            "auto_fallback": "使用 auto_chat, gpt-4, gpt-4.1, claude-* 等别名，自动回退到可用模型",
            "direct_call": "直接使用模型名称，不进行回退"
        },
        "fallback_order": ["QWEN", "OPENROUTER", "CEREBRAS", "CHUTES", "MINIMAX", "OLLAMA", "SUANLI"],
        "file_formats": {
            "openai": "file_url (new) or image_url (legacy)",
            "anthropic": "image/video/audio/document with url or base64"
        },
        "fixes": [
            "移除了有问题的 TimeoutManager 超时包装",
            "直接使用底层客户端的原生超时机制",
            "修复了首包延迟虚假超时问题",
            "优化了异步生成器的处理逻辑"
        ]
    })

@app.errorhandler(404)
async def not_found(error):
    return jsonify(create_error_response("not_found", "接口不存在", 404)), 404

@app.errorhandler(500)
async def internal_error(error):
    return jsonify(create_error_response("internal_error", "服务器内部错误", 500)), 500
