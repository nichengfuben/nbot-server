import requests
import time
import threading
from typing import Optional, Dict, Any
from common.logger import get_logger
logger = get_logger("tts")
import os

class TTSClient:
    """
    文本转语音客户端，支持多线程安全调用
    """
    
    def __init__(self, api_url: str = "https://ai.airoe.cn/v1/audio/speech"):
        self.api_url = api_url
        self._lock = threading.Lock()
        self.headers = {
            "Content-Type": "application/json"
        }
        
        # 支持的语音模型
        self.voices = {
            "思远": "male-botong",
            "心悦": "Podcast_girl", 
            "子轩": "boyan_new_hailuo",
            "灵儿": "female-shaonv",
            "语嫣": "YaeMiko_hailuo",
            "少泽": "xiaoyi_mix_hailuo",
            "芷溪": "xiaomo_sft",
            "浩翔": "cove_test2_hailuo",
            "雅涵": "scarlett_hailuo",
            "雷电将军": "Leishen2_hailuo",
            "钟离": "Zhongli_hailuo",
            "派蒙": "Paimeng_hailuo",
            "可莉": "keli_hailuo",
            "胡桃": "Hutao_hailuo",
            "熊二": "Xionger_hailuo",
            "海绵宝宝": "Haimian_hailuo",
            "变形金刚": "Robot_hunter_hailuo",
            "小玲玲": "Linzhiling_hailuo",
            "拽妃": "huafei_hailuo",
            "东北er": "lingfeng_hailuo",
            "老铁": "male_dongbei_hailuo",
            "北京er": "Beijing_hailuo",
            "JayJay": "JayChou_hailuo",
            "潇然": "Daniel_hailuo",
            "沉韵": "Bingjiao_zongcai_hailuo",
            "瑶瑶": "female-yaoyao-hd",
            "晨曦": "murong_sft",
            "沐珊": "shangshen_sft",
            "祁辰": "kongchen_sft",
            "夏洛特": "shenteng2_hailuo",
            "郭嘚嘚": "Guodegang_hailuo",
            "小月月": "yueyue_hailuo"
        }

    def text_to_speech(self, 
                      text: str, 
                      voice: str = "派蒙",
                      save_path: Optional[str] = None,
                      model: str = "tts") -> Dict[str, Any]:
        """
        将文本转换为语音文件
        
        Args:
            text: 要转换的文本
            voice: 语音模型名称（支持中文名称或英文model名）
            save_path: 保存路径，默认为当前目录下的output.mp3
            model: TTS模型类型
            
        Returns:
            包含结果信息的字典
        """
        
        with self._lock:  # 线程锁保护
            try:
                # 处理语音模型名称
                if voice in self.voices:
                    voice_model = self.voices[voice]
                else:
                    voice_model = voice  # 直接使用英文model名
                
                # 设置默认保存路径
                if save_path is None:
                    timestamp = int(time.time())
                    save_path = f"output_{timestamp}.mp3"
                
                # 确保目录存在
                os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
                
                # 构建请求数据
                payload = {
                    "model": model,
                    "input": text,
                    "voice": voice_model
                }
                
                # 记录开始时间
                start_time = time.time()
                
                # 发送请求
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    stream=True,
                    timeout=30
                )
                
                if response.status_code == 200:
                    # 保存音频文件
                    with open(save_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=1024):
                            if chunk:
                                f.write(chunk)
                    
                    # 计算耗时
                    duration = time.time() - start_time
                    
                    return {
                        "success": True,
                        "message": "音频生成成功",
                        "file_path": save_path,
                        "duration": round(duration, 2),
                        "voice_used": voice_model
                    }
                else:
                    return {
                        "success": False,
                        "message": f"请求失败，状态码: {response.status_code}",
                        "error": response.text
                    }
                    
            except Exception as e:
                return {
                    "success": False,
                    "message": f"发生错误: {str(e)}",
                    "error": str(e)
                }

    def get_available_voices(self) -> Dict[str, str]:
        """获取所有可用的语音模型"""
        return self.voices.copy()

    def is_voice_available(self, voice: str) -> bool:
        """检查语音模型是否可用"""
        return voice in self.voices or voice in self.voices.values()


# 全局客户端实例
_global_client = None
_client_lock = threading.Lock()

def get_tts_client() -> TTSClient:
    """获取全局TTS客户端实例（单例模式）"""
    global _global_client
    if _global_client is None:
        with _client_lock:
            if _global_client is None:
                _global_client = TTSClient()
    return _global_client

# 简洁的函数入口
def tts(text: str, voice: str = "派蒙", save_path: Optional[str] = None) -> Dict[str, Any]:
    """
    简洁的TTS函数入口
    
    Args:
        text: 要转换的文本
        voice: 语音模型（默认派蒙）
        save_path: 保存路径
        
    Returns:
        结果字典
        
    Example:
        result = tts("你好世界", "派蒙", "hello.mp3")
        if result["success"]:
            print(f"音频已保存到: {result['file_path']}")
    """
    client = get_tts_client()
    return client.text_to_speech(text, voice, save_path)

def get_voices() -> Dict[str, str]:
    """获取所有可用语音"""
    client = get_tts_client()
    return client.get_available_voices()

if __name__ == "__main__":
    # 测试代码
    result = tts("现在几点了", "派蒙", "test.mp3")
    logger.info(result)
    
    # 显示可用语音
    voices = get_voices()
    logger.info("\n可用语音:")
    for chinese, english in voices.items():
        logger.info(f"{chinese}: {english}")
