# embed_client.py
import os
import requests
import json
from typing import List, Optional, Dict, Any
import sys

# 添加common目录到路径以便导入logger
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.logger import get_logger
logger = get_logger("embed")

class EmbedClient:
    def __init__(self, 
                 model: str = 'Qwen/Qwen3-Embedding-8B', 
                 base_url: str = "https://ai.airoe.cn/v1",
                 api_key: str = "sk-IuErtfaMgq2XPBXwlSYzD34Yr0W0y8NRXh025FnmSIceXSx9",
                 dimensions: int = 1024):
        """
        初始化嵌入向量客户端，仅支持在线API
        
        Args:
            model: 在线API使用的模型名称
            base_url: 在线API服务的基础URL
            api_key: 在线API密钥
            dimensions: 返回向量的维度，默认1024
        """
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.dimensions = dimensions
        
        # 服务状态
        self.service_available = True
    
    def _check_online_service(self) -> bool:
        """检查在线API服务状态"""
        try:
            test_embedding = self._get_online_embedding("test")
            return len(test_embedding) > 0
        except Exception as e:
            logger.debug(f"在线服务状态检查失败: {e}")
            return False
    
    def _get_online_embedding(self, text: str, dimensions: Optional[int] = None) -> List[float]:
        """从在线API获取嵌入向量"""
        try:
            url = f"{self.base_url}/embeddings"
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            dim = dimensions if dimensions is not None else self.dimensions
            
            payload = {
                "model": self.model,
                "input": text
            }
            
            # 只有在指定了dimensions时才添加该参数
            if dim is not None:
                payload["dimensions"] = dim
            
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            
            # 从响应中提取嵌入向量
            if 'data' in result and len(result['data']) > 0:
                embedding = result['data'][0].get('embedding', [])
                logger.debug(f"从在线API获取到 {len(embedding)} 维向量")
                return embedding
            else:
                logger.error(f"在线API响应格式异常: {result}")
                return []
            
        except Exception as e:
            logger.debug(f"在线API请求失败: {e}")
            raise
    
    def get_embedding(self, text: str, dimensions: Optional[int] = None) -> List[float]:
        """
        获取单个文本的嵌入向量
        
        Args:
            text: 输入文本
            dimensions: 返回向量的维度，覆盖实例的默认设置
            
        Returns:
            嵌入向量列表
        """
        if not self.service_available:
            logger.warning("没有可用的嵌入向量服务")
            return []
        
        dim = dimensions if dimensions is not None else self.dimensions
        
        try:
            embedding = self._get_online_embedding(text, dimensions=dim)
            return embedding
        except Exception as e:
            logger.error(f"在线API失败: {e}")
            return []
    
    def get_embeddings_batch(self, texts: List[str], delay: float = 0.1, dimensions: Optional[int] = None) -> List[List[float]]:
        """
        批量获取多个文本的嵌入向量
        
        Args:
            texts: 文本列表
            delay: 请求间隔时间（秒），避免请求过快
            dimensions: 返回向量的维度，覆盖实例的默认设置
            
        Returns:
            嵌入向量列表的列表
        """
        embeddings = []
        for i, text in enumerate(texts):
            embedding = self.get_embedding(text, dimensions=dimensions)
            embeddings.append(embedding)
            
            # 添加小延迟避免过快请求（最后一个不需要延迟）
            if i < len(texts) - 1:
                time.sleep(delay)
        
        return embeddings
    
    def similarity(self, text1: str, text2: str, dimensions: Optional[int] = None) -> float:
        """
        计算两个文本的余弦相似度（使用numpy）
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            dimensions: 返回向量的维度，覆盖实例的默认设置
            
        Returns:
            相似度分数 (-1到1之间)
        """
        try:
            import numpy as np
            
            embedding1 = self.get_embedding(text1, dimensions=dimensions)
            embedding2 = self.get_embedding(text2, dimensions=dimensions)
            
            if not embedding1 or not embedding2:
                logger.warning("无法获取嵌入向量")
                return 0.0
            
            # 计算余弦相似度
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # 避免除零错误
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                logger.warning("向量范数为0")
                return 0.0
            
            cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
            return float(cosine_sim)
        
        except ImportError:
            logger.error("需要安装 numpy: pip install numpy")
            logger.info("使用 similarity_without_numpy 方法")
            return self.similarity_without_numpy(text1, text2, dimensions=dimensions)
        except Exception as e:
            logger.error(f"计算相似度时出错: {e}")
            return 0.0
    
    def similarity_without_numpy(self, text1: str, text2: str, dimensions: Optional[int] = None) -> float:
        """
        不使用numpy计算两个文本的余弦相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            dimensions: 返回向量的维度，覆盖实例的默认设置
            
        Returns:
            相似度分数 (-1到1之间)
        """
        try:
            embedding1 = self.get_embedding(text1, dimensions=dimensions)
            embedding2 = self.get_embedding(text2, dimensions=dimensions)
            
            if not embedding1 or not embedding2 or len(embedding1) != len(embedding2):
                logger.warning("嵌入向量无效或长度不匹配")
                return 0.0
            
            # 手动计算余弦相似度
            dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
            norm1 = sum(a * a for a in embedding1) ** 0.5
            norm2 = sum(b * b for b in embedding2) ** 0.5
            
            if norm1 == 0 or norm2 == 0:
                logger.warning("向量范数为0")
                return 0.0
            
            cosine_sim = dot_product / (norm1 * norm2)
            return float(cosine_sim)
        
        except Exception as e:
            logger.error(f"计算相似度时出错: {e}")
            return 0.0
    
    def check_service_status(self) -> bool:
        """检查服务状态"""
        return self.service_available
    
    def get_service_info(self) -> Dict[str, Any]:
        """获取当前服务信息"""
        return {
            "service_available": self.service_available,
            "current_service": "online",
            "dimensions": self.dimensions,
        }
    
    def get_embedding_info(self, text: str, dimensions: Optional[int] = None) -> Dict[str, Any]:
        """
        获取嵌入向量的详细信息
        
        Args:
            text: 输入文本
            dimensions: 返回向量的维度，覆盖实例的默认设置
            
        Returns:
            包含嵌入向量和元信息的字典
        """
        embedding = self.get_embedding(text, dimensions=dimensions)
        dim = dimensions if dimensions is not None else self.dimensions
        
        return {
            "embedding": embedding,
            "model": self.model,
            "service": "online",
            "dimension": len(embedding),
            "requested_dimension": dim,
            "service_available": self.service_available
        }
    
    def set_dimensions(self, dimensions: int):
        """
        设置默认的向量维度
        
        Args:
            dimensions: 向量维度
        """
        self.dimensions = dimensions
        logger.info(f"默认向量维度已设置为: {dimensions}")


# 使用示例
if __name__ == "__main__":
    import time
    logger.info("=" * 60)
    logger.info("测试嵌入向量客户端（仅在线API）")
    logger.info("=" * 60)
    
    # 创建客户端（默认使用1024维度）
    client = EmbedClient(dimensions=1024)
    
    # 检查服务状态
    service_info = client.get_service_info()
    logger.info(f"服务状态: {'可用' if service_info['service_available'] else '不可用'}")
    logger.info(f"当前使用: {service_info['current_service']}服务")
    logger.info(f"默认维度: {service_info['dimensions']}")
    logger.info('')
    
    if not client.check_service_status():
        logger.info("嵌入向量服务不可用")
        exit()
    
    # 获取单个嵌入向量
    logger.info("=== 测试单个嵌入向量（1024维） ===")
    embedding = client.get_embedding("这是一段测试文本，用于生成向量嵌入")
    logger.info(f"嵌入向量维度: {len(embedding)}")
    if embedding:
        logger.info(f"前5个向量值: {embedding[:5]}")
        logger.info(f"使用的服务: {client.get_service_info()['current_service']}\n")
    
    logger.info("=" * 60)
    logger.info("测试 2: 批量处理")
    logger.info("=" * 60)
    texts = ["你好", "Hello", "再见", "Goodbye"]
    embeddings = client.get_embeddings_batch(texts)
    logger.info(f"批量处理了 {len(embeddings)} 个文本")
    for i, text in enumerate(texts):
        logger.info(f"  {text}: 向量维度 {len(embeddings[i])}")
    logger.info('')
    
    logger.info("=" * 60)
    logger.info("测试 3: 相似度计算")
    logger.info("=" * 60)
    
    similarity_score = client.similarity_without_numpy("你好", "Hello")
    logger.info(f"'你好' 和 'Hello' 的相似度: {similarity_score:.4f}")
    
    similarity_score2 = client.similarity_without_numpy("苹果", "水果")
    logger.info(f"'苹果' 和 '水果' 的相似度: {similarity_score2:.4f}")
    
    similarity_score3 = client.similarity_without_numpy("电脑", "篮球")
    logger.info(f"'电脑' 和 '篮球' 的相似度: {similarity_score3:.4f}")
    logger.info('')
    
    logger.info("=" * 60)
    logger.info("测试 4: 获取详细信息")
    logger.info("=" * 60)
    info = client.get_embedding_info("测试获取详细信息")
    logger.info(f"模型: {info['model']}")
    logger.info(f"服务: {info['service']}")
    logger.info(f"请求维度: {info['requested_dimension']}")
    logger.info(f"实际维度: {info['dimension']}")
    logger.info(f"服务可用: {info['service_available']}")
    
    logger.info("\n" + "=" * 60)
    logger.info("测试 5: 动态修改维度")
    logger.info("=" * 60)
    client.set_dimensions(512)
    embedding_512 = client.get_embedding("测试修改维度")
    logger.info(f"修改后的维度: {len(embedding_512)}")
    
    # 临时指定不同维度
    embedding_256 = client.get_embedding("测试临时维度", dimensions=256)
    logger.info(f"临时指定256维: {len(embedding_256)}")
