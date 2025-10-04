#api/index.py
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import json
import uuid
import time
import tempfile
import os
import threading
from typing import *
from concurrent.futures import ThreadPoolExecutor
import logging
import requests
import base64
from queue import Queue, Empty
import asyncio

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

app = Flask(__name__)
CORS(app)


# ==================== 异步事件循环管理器 ====================
class AsyncLoopManager:
    """全局异步事件循环管理器 - 解决Flask中的asyncio问题"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self.loop = None
        self.thread = None
        self._start_loop()
    
    def _start_loop(self):
        """在独立线程中启动事件循环"""
        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()
        
        self.thread = threading.Thread(target=run_loop, daemon=True)
        self.thread.start()
        
        # 等待loop初始化
        while self.loop is None:
            time.sleep(0.01)
    
    def run_coroutine(self, coro, timeout=300):
        """在事件循环中运行协程并返回结果"""
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        try:
            return future.result(timeout=timeout)
        except Exception as e:
            raise e
    
    def run_async_generator(self, async_gen, queue, stop_event):
        """在事件循环中运行异步生成器，输出到队列"""
        async def consume():
            try:
                async for item in async_gen:
                    if stop_event.is_set():
                        break
                    queue.put(('data', item))
            except Exception as e:
                queue.put(('error', str(e)))
            finally:
                queue.put(('done', None))
        
        asyncio.run_coroutine_threadsafe(consume(), self.loop)
    
    def stop(self):
        """停止事件循环"""
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)


# 全局异步循环管理器
async_manager = AsyncLoopManager()


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
        "max_files": 20,
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
        "context_length": 65536,
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
        "max_files": 1,
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
    def download_file(url: str, temp_dir: str) -> str:
        """下载网络文件到本地临时文件"""
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                content = response.content
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
    """统一客户端处理器 - Flask同步版本（使用全局事件循环）"""
    
    def __init__(self, max_concurrent: int = 100):
        self.max_concurrent = max_concurrent
        self.chat_semaphore = threading.Semaphore(max_concurrent)
        self.embed_semaphore = threading.Semaphore(10)
        self.download_semaphore = threading.Semaphore(20)
        
        self.thread_pool = ThreadPoolExecutor(max_workers=50)
        
        self.ollama_client = None
        self.suanli_client = None
        self.embed_client = None
        self._client_lock = threading.Lock()
        
        self.temp_dir = tempfile.mkdtemp()
        self.temp_files = set()
        self._temp_lock = threading.Lock()
        
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'model_usage': {config['name']: 0 for config in ModelConfig.FALLBACK_ORDER},
            'fallback_usage': 0,
            'file_downloads': 0,
            'multimodal_rejections': 0,
            'direct_model_calls': 0
        }
    
    def _get_ollama_client(self) -> OllamaClient:
        """获取Ollama客户端实例（单例模式）"""
        if self.ollama_client is None:
            with self._client_lock:
                if self.ollama_client is None:
                    self.ollama_client = OllamaClient()
        return self.ollama_client
    
    def _get_suanli_client(self) -> SuanliClient:
        """获取Suanli客户端实例（单例模式）"""
        if self.suanli_client is None:
            with self._client_lock:
                if self.suanli_client is None:
                    self.suanli_client = SuanliClient()
        return self.suanli_client
    
    def _get_embed_client(self) -> EmbedClient:
        """获取嵌入客户端实例（单例模式）"""
        if self.embed_client is None:
            with self._client_lock:
                if self.embed_client is None:
                    self.embed_client = EmbedClient()
        return self.embed_client
    
    def _add_temp_file(self, file_path: str) -> None:
        """添加临时文件到管理列表"""
        with self._temp_lock:
            self.temp_files.add(file_path)
    
    def _remove_temp_file(self, file_path: str) -> None:
        """从管理列表移除临时文件"""
        with self._temp_lock:
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
        """验证模型是否能处理指定的文件"""
        if not file_paths:
            return True, ""
        
        if len(file_paths) > model_config.get('max_files', 0):
            if model_config['name'] == 'QWEN':
                return False, f"文件数量超限: {len(file_paths)} > {model_config['max_files']}"
            else:
                return False, f"模型 {model_config['name']} 不支持多文件，当前有 {len(file_paths)} 个文件"
        
        file_analysis = FileProcessor.analyze_files(file_paths)
        
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
        """判断是否应该将长文本转换为文档处理"""
        if (len(text) > model_config['context_length'] and 
            model_config.get('supports_document') and 
            len(text) <= model_config.get('max_doc_chars', 0)):
            
            doc_path = FileProcessor.text_to_temp_document(text, model_config.get('max_doc_chars'))
            self._add_temp_file(doc_path)
            return True, doc_path
        return False, None
    
    def _try_qwen(self, text: str, file_paths: Optional[List[str]] = None, stream: bool = False):
        """尝试使用QWEN模型"""
        config = ModelConfig.QWEN
        should_doc, doc_path = self._should_handle_as_document(text, config)
        if should_doc:
            file_paths = [doc_path] if not file_paths else (file_paths + [doc_path])
        else:
            text = self._truncate_text(text, config['context_length'])
        
        try:
            if stream:
                # 返回异步生成器
                return async_manager.run_coroutine(quick_stream(text, file_paths))
            else:
                return async_manager.run_coroutine(quick_chat(text, file_paths))
        except Exception as e:
            raise e
    
    def _try_openrouter(self, text: str, file_paths: Optional[List[str]] = None, stream: bool = False):
        """尝试使用OPENROUTER模型"""
        config = ModelConfig.OPENROUTER
        text = self._truncate_text(text, config['context_length'])
        
        image_path = ""
        if file_paths:
            file_analysis = FileProcessor.analyze_files(file_paths)
            if file_analysis['images']:
                image_path = file_analysis['images'][0]
        
        try:
            if stream:
                return async_manager.run_coroutine(openrouter_stream(text, image_path))
            else:
                return async_manager.run_coroutine(openrouter_chat(text, image_path))
        except Exception as e:
            raise e
    
    def _try_cerebras(self, text: str, stream: bool = False):
        """尝试使用CEREBRAS模型"""
        config = ModelConfig.CEREBRAS
        text = self._truncate_text(text, config['context_length'])
        
        if stream:
            return async_manager.run_coroutine(cerebras_stream(text))
        else:
            return async_manager.run_coroutine(cerebras_chat(text))
    
    def _try_chutes(self, text: str, stream: bool = False):
        """尝试使用CHUTES模型"""
        config = ModelConfig.CHUTES
        text = self._truncate_text(text, config['context_length'])
        
        if stream:
            return async_manager.run_coroutine(chutes_stream(text))
        else:
            return async_manager.run_coroutine(chutes_chat(text))
    
    def _try_minimax(self, text: str, file_paths: Optional[List[str]] = None, stream: bool = False):
        """尝试使用MINIMAX模型"""
        config = ModelConfig.MINIMAX
        text = self._truncate_text(text, config['context_length'])
        
        image_path = None
        if file_paths:
            file_analysis = FileProcessor.analyze_files(file_paths)
            if file_analysis['images']:
                image_path = file_analysis['images'][0]
        
        if stream:
            return async_manager.run_coroutine(minimax_stream(text, image_path))
        else:
            return async_manager.run_coroutine(minimax_chat(text, image_path))
    
    def _try_ollama(self, text: str, stream: bool = False):
        """尝试使用OLLAMA模型"""
        config = ModelConfig.OLLAMA
        text = self._truncate_text(text, config['context_length'])
        
        client = self._get_ollama_client()
        
        if stream:
            return async_manager.run_coroutine(client.chat_stream(text))
        else:
            return async_manager.run_coroutine(client.chat(text))
    
    def _try_suanli(self, text: str, stream: bool = False):
        """尝试使用SUANLI模型"""
        config = ModelConfig.SUANLI
        text = self._truncate_text(text, config['context_length'])
        
        client = self._get_suanli_client()
        
        if stream:
            def suanli_stream_generator():
                for chunk in client.chat_stream_generator(text, show_stats=False):
                    if not chunk.startswith("❌"):
                        yield chunk
            
            return suanli_stream_generator()
        else:
            result, _ = client.chat(text, show_stats=False)
            if result is None:
                raise Exception("Suanli返回None")
            return result
    
    def _execute_with_fallback(self, text: str, file_paths: Optional[List[str]] = None, stream: bool = False, retries: int = 0):
        """执行带回退逻辑的模型调用"""
        with self.chat_semaphore:
            self.stats['total_requests'] += 1
            
            processed_files = []
            if file_paths:
                try:
                    processed_files = file_paths
                except Exception as e:
                    raise Exception(f"文件处理失败: {str(e)}")
            
            for attempt in range(retries + 1):
                last_error = None
                
                for config in ModelConfig.FALLBACK_ORDER:
                    can_handle, error_msg = self._validate_files_for_model(config, processed_files)
                    
                    if not can_handle:
                        if config['name'] == 'QWEN':
                            self.stats['multimodal_rejections'] += 1
                            raise Exception(error_msg)
                        continue
                    
                    try:
                        if config['name'] == 'QWEN':
                            result = self._try_qwen(text, processed_files, stream)
                        elif config['name'] == 'OPENROUTER':
                            result = self._try_openrouter(text, processed_files, stream)
                        elif config['name'] == 'CEREBRAS':
                            result = self._try_cerebras(text, stream)
                        elif config['name'] == 'CHUTES':
                            result = self._try_chutes(text, stream)
                        elif config['name'] == 'MINIMAX':
                            result = self._try_minimax(text, processed_files, stream)
                        elif config['name'] == 'OLLAMA':
                            result = self._try_ollama(text, stream)
                        elif config['name'] == 'SUANLI':
                            result = self._try_suanli(text, stream)
                        
                        self.stats['successful_requests'] += 1
                        self.stats['model_usage'][config['name']] += 1
                        return result
                        
                    except Exception as e:
                        last_error = e
                        logger.warning(f"模型 {config['name']} 调用失败 (尝试 {attempt + 1}): {str(e)}")
                        continue
                
                if attempt < retries:
                    time.sleep(1)
            
            self.stats['fallback_usage'] += 1
            error_message = f"所有模型调用失败，最后一个错误: {str(last_error)}" if last_error else "所有模型调用失败"
            raise Exception(error_message)
    
    def chat_with_specific_model(self, model_name: str, text: str, file_paths: Optional[List[str]] = None, stream: bool = False, retries: int = 0):
        """调用指定的模型（不使用回退机制）"""
        with self.chat_semaphore:
            self.stats['total_requests'] += 1
            self.stats['direct_model_calls'] += 1
            
            model_name = model_name.upper()
            
            model_configs = {config['name']: config for config in ModelConfig.FALLBACK_ORDER}
            if model_name not in model_configs:
                raise Exception(f"不支持的模型: {model_name}")
            
            config = model_configs[model_name]
            
            can_handle, error_msg = self._validate_files_for_model(config, file_paths or [])
            if not can_handle:
                raise Exception(error_msg)
            
            processed_files = file_paths or []
            
            for attempt in range(retries + 1):
                try:
                    if model_name == 'QWEN':
                        result = self._try_qwen(text, processed_files, stream)
                    elif model_name == 'OPENROUTER':
                        result = self._try_openrouter(text, processed_files, stream)
                    elif model_name == 'CEREBRAS':
                        result = self._try_cerebras(text, stream)
                    elif model_name == 'CHUTES':
                        result = self._try_chutes(text, stream)
                    elif model_name == 'MINIMAX':
                        result = self._try_minimax(text, processed_files, stream)
                    elif model_name == 'OLLAMA':
                        result = self._try_ollama(text, stream)
                    elif model_name == 'SUANLI':
                        result = self._try_suanli(text, stream)
                    else:
                        raise Exception(f"未实现的模型调用: {model_name}")
                    
                    self.stats['successful_requests'] += 1
                    self.stats['model_usage'][model_name] += 1
                    return result
                    
                except Exception as e:
                    if attempt < retries:
                        time.sleep(1)
                        continue
                    raise e
    
    def chat_with_model_stream(self, text: str, file_paths: Optional[List[str]] = None, retries: int = 0):
        """流式聊天接口（使用回退机制）"""
        result = self._execute_with_fallback(text, file_paths, stream=True, retries=retries)
        
        # 检查是否是异步生成器
        if hasattr(result, '__aiter__'):
            # 异步生成器，使用队列桥接
            queue = Queue(maxsize=100)
            stop_event = threading.Event()
            async_manager.run_async_generator(result, queue, stop_event)
            
            while True:
                try:
                    msg_type, data = queue.get(timeout=60)
                    if msg_type == 'done':
                        break
                    elif msg_type == 'error':
                        raise Exception(data)
                    elif msg_type == 'data':
                        yield data
                except Empty:
                    stop_event.set()
                    raise Exception("Stream timeout")
        else:
            # 同步生成器
            for chunk in result:
                yield chunk
    
    def chat_with_model(self, text: str, file_paths: Optional[List[str]] = None, retries: int = 0) -> str:
        """非流式聊天接口（使用回退机制）"""
        result = self._execute_with_fallback(text, file_paths, stream=False, retries=retries)
        return result
    
    def chat_with_tts(self, text: str, voice: str = "派蒙", save_path: Optional[str] = None, retries: int = 0) -> Dict[str, Any]:
        """文本转语音接口"""
        for attempt in range(retries + 1):
            try:
                result = tts(text, voice, save_path)
                return result
            except Exception as e:
                if attempt < retries:
                    time.sleep(1)
                    continue
                raise e
    
    def chat_with_embed(self, text: str, retries: int = 0) -> List[float]:
        """文本嵌入接口"""
        logger.info(text)
        with self.embed_semaphore:
            for attempt in range(retries + 1):
                try:
                    client = self._get_embed_client()
                    embedding = client.get_embedding(text)
                    return embedding
                except Exception as e:
                    if attempt < retries:
                        time.sleep(1)
                        continue
                    raise e
    
    def cleanup_temp_files(self, file_paths: List[str]) -> None:
        """清理临时文件"""
        for file_path in file_paths:
            self._remove_temp_file(file_path)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        success_rate = (self.stats['successful_requests'] / max(self.stats['total_requests'], 1)) * 100
        return {
            **self.stats,
            'success_rate': round(success_rate, 2),
            'fallback_rate': round((self.stats['fallback_usage'] / max(self.stats['total_requests'], 1)) * 100, 2),
            'direct_call_rate': round((self.stats['direct_model_calls'] / max(self.stats['total_requests'], 1)) * 100, 2)
        }
    
    def close(self) -> None:
        """清理资源"""
        self.thread_pool.shutdown(wait=True)
        
        with self._temp_lock:
            for file_path in list(self.temp_files):
                self._remove_temp_file(file_path)
            
            try:
                os.rmdir(self.temp_dir)
            except:
                pass
        
        try:
            if qwen_cleanup:
                async_manager.run_coroutine(qwen_cleanup())
        except:
            pass


# 全局处理器实例
_global_handler = None
_handler_lock = threading.Lock()


def get_handler() -> ClientHandler:
    """获取全局处理器实例（单例模式）"""
    global _global_handler
    if _global_handler is None:
        with _handler_lock:
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


def create_openai_stream_response(content_generator, model: str):
    """创建OpenAI格式的流式响应"""
    yield f"data: {json.dumps({'id': f'chatcmpl-{uuid.uuid4().hex}', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
    
    for chunk in content_generator:
        if chunk:
            yield f"data: {json.dumps({'id': f'chatcmpl-{uuid.uuid4().hex}', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model, 'choices': [{'index': 0, 'delta': {'content': chunk}, 'finish_reason': None}]})}\n\n"
    
    yield f"data: {json.dumps({'id': f'chatcmpl-{uuid.uuid4().hex}', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
    yield "data: [DONE]\n\n"


def create_anthropic_stream_response(content_generator, model: str):
    """创建Anthropic格式的流式响应"""
    message_id = f"msg_{uuid.uuid4().hex}"
    
    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': message_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': model, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
    
    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
    
    for chunk in content_generator:
        if chunk:
            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': chunk}})}\n\n"
    
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
                    if item.get('type') == 'file_url':
                        file_url = item.get('file_url', {}).get('url')
                        if file_url:
                            file_urls.append(file_url)
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
                    
                    if item_type in ['image', 'video', 'audio', 'document']:
                        source = item.get('source', {})
                        if source.get('type') == 'url':
                            url = source.get('url')
                            if url:
                                file_urls.append(url)
                        elif source.get('type') == 'base64':
                            media_type = source.get('media_type', '')
                            data = source.get('data', '')
                            if data:
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
def chat_completions():
    """OpenAI兼容的聊天完成API"""
    temp_files = []
    try:
        data = request.get_json()
        if not data or 'messages' not in data:
            return jsonify(create_error_response("invalid_request", "缺少messages参数", 400)), 400
        
        model = data.get('model', 'auto_chat')
        messages = data.get('messages', [])
        stream = data.get('stream', False)
        
        options = data.get('options', {})
        retries = options.get('retries', 0)
        
        text_content = extract_text_from_messages(messages, "openai")
        
        if not text_content.strip():
            return jsonify(create_error_response("invalid_request", "消息内容为空", 400)), 400
        
        handler = get_handler()
        
        file_urls = []
        files = request.files
        if files:
            for file_key in files:
                file = files[file_key]
                if file.filename:
                    file_path = os.path.join(handler.temp_dir, f"{uuid.uuid4().hex}_{file.filename}")
                    file.save(file_path)
                    handler._add_temp_file(file_path)
                    file_urls.append(file_path)
                    temp_files.append(file_path)
        
        message_files = extract_files_from_messages(messages, "openai")
        file_urls.extend(message_files)
        temp_files.extend([f for f in message_files if not f.startswith(('http://', 'https://'))])
        
        model_configs = {config['name']: config for config in ModelConfig.FALLBACK_ORDER}
        model_upper = model.upper()
        
        try:
            if model_upper in model_configs:
                if stream:
                    def generate():
                        try:
                            for chunk in create_openai_stream_response(
                                handler.chat_with_specific_model(model_upper, text_content, file_urls or None, True, retries), 
                                model
                            ):
                                yield chunk
                        except Exception as e:
                            error_chunk = f"data: {json.dumps(create_error_response('api_error', str(e)))}\n\n"
                            yield error_chunk
                            yield "data: [DONE]\n\n"
                        finally:
                            handler.cleanup_temp_files(temp_files)
                    
                    return Response(stream_with_context(generate()), mimetype='text/event-stream')
                else:
                    content = handler.chat_with_specific_model(model_upper, text_content, file_urls or None, False, retries)
                    handler.cleanup_temp_files(temp_files)
                    return jsonify(create_openai_response(content, model))
            
            elif model in ['auto_chat', 'gpt-4', 'gpt-4.1'] or model.startswith("claude"):
                if stream:
                    def generate():
                        try:
                            for chunk in create_openai_stream_response(
                                handler.chat_with_model_stream(text_content, file_urls or None, retries), 
                                model
                            ):
                                yield chunk
                        except Exception as e:
                            error_chunk = f"data: {json.dumps(create_error_response('api_error', str(e)))}\n\n"
                            yield error_chunk
                            yield "data: [DONE]\n\n"
                        finally:
                            handler.cleanup_temp_files(temp_files)
                    
                    return Response(stream_with_context(generate()), mimetype='text/event-stream')
                else:
                    content = handler.chat_with_model(text_content, file_urls or None, retries)
                    handler.cleanup_temp_files(temp_files)
                    return jsonify(create_openai_response(content, model))
            
            elif model == 'auto_tts':
                try:
                    voice = data.get('voice', '派蒙')
                    result = handler.chat_with_tts(text_content, voice, retries=retries)
                    handler.cleanup_temp_files(temp_files)
                    return jsonify(result)
                except Exception as e:
                    handler.cleanup_temp_files(temp_files)
                    return jsonify(create_error_response("tts_error", str(e), 500)), 500
            
            elif model == 'auto_embedding':
                try:
                    embedding = handler.chat_with_embed(text_content, retries=retries)
                    handler.cleanup_temp_files(temp_files)
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
                    handler.cleanup_temp_files(temp_files)
                    return jsonify(create_error_response("embedding_error", str(e), 500)), 500
            
            else:
                handler.cleanup_temp_files(temp_files)
                return jsonify(create_error_response("invalid_model", f"不支持的模型: {model}", 400)), 400
        
        except Exception as e:
            handler.cleanup_temp_files(temp_files)
            return jsonify(create_error_response("api_error", str(e), 500)), 500
    
    except Exception as e:
        logger.error(f"OpenAI API调用出错: {str(e)}")
        return jsonify(create_error_response("internal_error", str(e), 500)), 500


@app.route('/v1/messages', methods=['POST'])
def anthropic_messages():
    """Anthropic兼容的消息API"""
    temp_files = []
    try:
        data = request.get_json()

        if not data or 'messages' not in data:
            return jsonify(create_error_response("invalid_request", "缺少messages参数", 400)), 400
        
        model = data.get('model', 'auto_chat')
        messages = data.get('messages', [])
        stream = data.get('stream', False)

        options = data.get('options', {})
        retries = options.get('retries', 0)
        
        text_content = extract_text_from_messages(messages, "anthropic")
        
        if not text_content.strip():
            return jsonify(create_error_response("invalid_request", "消息内容为空", 400)), 400
        
        handler = get_handler()
        
        file_urls = []
        files = request.files
        if files:
            for file_key in files:
                file = files[file_key]
                if file.filename:
                    file_path = os.path.join(handler.temp_dir, f"{uuid.uuid4().hex}_{file.filename}")
                    file.save(file_path)
                    handler._add_temp_file(file_path)
                    file_urls.append(file_path)
                    temp_files.append(file_path)
        
        message_files = extract_files_from_messages(messages, "anthropic")
        file_urls.extend(message_files)
        temp_files.extend([f for f in message_files if not f.startswith(('http://', 'https://'))])
        
        model_configs = {config['name']: config for config in ModelConfig.FALLBACK_ORDER}
        model_upper = model.upper()
        
        try:
            if model_upper in model_configs:
                if stream:
                    def generate():
                        try:
                            for chunk in create_anthropic_stream_response(
                                handler.chat_with_specific_model(model_upper, text_content, file_urls or None, True, retries), 
                                model
                            ):
                                yield chunk
                        except Exception as e:
                            error_chunk = f"event: error\ndata: {json.dumps(create_error_response('api_error', str(e)))}\n\n"
                            yield error_chunk
                        finally:
                            handler.cleanup_temp_files(temp_files)
                    
                    return Response(stream_with_context(generate()), mimetype='text/event-stream')
                else:
                    content = handler.chat_with_specific_model(model_upper, text_content, file_urls or None, False, retries)
                    handler.cleanup_temp_files(temp_files)
                    return jsonify(create_anthropic_response(content, model))
            
            elif model.startswith("claude") or model == "auto_chat":
                if stream:
                    def generate():
                        try:
                            for chunk in create_anthropic_stream_response(
                                handler.chat_with_model_stream(text_content, file_urls or None, retries), 
                                model
                            ):
                                yield chunk
                        except Exception as e:
                            error_chunk = f"event: error\ndata: {json.dumps(create_error_response('api_error', str(e)))}\n\n"
                            yield error_chunk
                        finally:
                            handler.cleanup_temp_files(temp_files)
                    
                    return Response(stream_with_context(generate()), mimetype='text/event-stream')
                else:
                    content = handler.chat_with_model(text_content, file_urls or None, retries)
                    handler.cleanup_temp_files(temp_files)
                    return jsonify(create_anthropic_response(content, model))
            
            else:
                handler.cleanup_temp_files(temp_files)
                return jsonify(create_error_response("invalid_model", f"不支持的模型: {model}", 400)), 400
        
        except Exception as e:
            handler.cleanup_temp_files(temp_files)
            return jsonify(create_error_response("api_error", str(e), 500)), 500
    
    except Exception as e:
        logger.error(f"Anthropic API调用出错: {str(e)}")
        return jsonify(create_error_response("internal_error", str(e), 500)), 500


@app.route('/v1/models', methods=['GET'])
def list_models():
    """列出可用模型"""
    models = [
        {"id": "auto_chat", "object": "model", "created": int(time.time()), "owned_by": "nbot"},
        {"id": "QWEN", "object": "model", "created": int(time.time()), "owned_by": "nbot"},
        {"id": "OPENROUTER", "object": "model", "created": int(time.time()), "owned_by": "nbot"},
        {"id": "CEREBRAS", "object": "model", "created": int(time.time()), "owned_by": "nbot"},
        {"id": "CHUTES", "object": "model", "created": int(time.time()), "owned_by": "nbot"},
        {"id": "MINIMAX", "object": "model", "created": int(time.time()), "owned_by": "nbot"},
        {"id": "OLLAMA", "object": "model", "created": int(time.time()), "owned_by": "nbot"},
        {"id": "SUANLI", "object": "model", "created": int(time.time()), "owned_by": "nbot"},
        {"id": "claude-3-sonnet-20240229", "object": "model", "created": int(time.time()), "owned_by": "nbot"},
        {"id": "claude-3-opus-20240229", "object": "model", "created": int(time.time()), "owned_by": "nbot"},
        {"id": "claude-3-haiku-20240307", "object": "model", "created": int(time.time()), "owned_by": "nbot"},
        {"id": "gpt-4", "object": "model", "created": int(time.time()), "owned_by": "nbot"},
        {"id": "gpt-4.1", "object": "model", "created": int(time.time()), "owned_by": "nbot"},
        {"id": "auto_tts", "object": "model", "created": int(time.time()), "owned_by": "nbot"},
        {"id": "auto_embedding", "object": "model", "created": int(time.time()), "owned_by": "nbot"}
    ]
    
    return jsonify({"object": "list", "data": models})


@app.route('/v1/health', methods=['GET'])
def health_check():
    """健康检查"""
    handler = get_handler()
    stats = handler.get_stats()
    return jsonify({
        "status": "healthy",
        "timestamp": int(time.time()),
        "stats": stats
    })


@app.route('/', methods=['GET'])
def index():
    """API文档"""
    return jsonify({
        "message": "OpenAI & Anthropic Compatible API (Flask + AsyncLoopManager)",
        "version": "2.2.2-flask-fixed",
        "framework": "Flask with dedicated asyncio event loop",
        "endpoints": {
            "openai_chat": "/v1/chat/completions",
            "anthropic_messages": "/v1/messages",
            "models": "/v1/models",
            "health": "/v1/health"
        }
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify(create_error_response("not_found", "接口不存在", 404)), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify(create_error_response("internal_error", "服务器内部错误", 500)), 500

handler = app

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
