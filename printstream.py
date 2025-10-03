import threading
import time
import math
import sys
import atexit
from typing import Any
from collections import deque

class PrintStream:
    """动态速度打印流系统 - 有序队列版本"""
    def __init__(self):
        # 使用队列保证文本顺序
        self.text_queue = deque()  # 存储待打印的文本块
        self.current_text = ""  # 当前正在打印的文本
        self.lock = threading.Lock()
        self.running = False
        self.output_thread = None
        
        # 速度控制参数
        self.min_speed = 5.0
        self.max_speed = 100.0
        self.decay_factor = 20.0
        self.smoothing_factor = 0.8
        self.current_speed = self.min_speed
        self.accumulated_chars = 0.0
        self._started = False
        
        # 统计信息
        self.total_pending_chars = 0  # 总待打印字符数

    def start(self):
        """启动打印流系统"""
        if not self.running and not self._started:
            self.running = True
            self._started = True
            self.output_thread = threading.Thread(target=self._output_processor, daemon=True)
            self.output_thread.start()

    def stop(self):
        """停止打印流系统"""
        if self.running:
            self.running = False
            # 等待所有内容输出完毕
            max_wait = 10.0  # 最多等待10秒
            start_time = time.time()
            while (self.current_text or self.text_queue) and (time.time() - start_time) < max_wait:
                time.sleep(0.1)
            
            if self.output_thread and self.output_thread.is_alive():
                self.output_thread.join(timeout=1)

    def add_to_buffer(self, text: str):
        """添加文本到队列"""
        if not self.running:
            self.start()
        
        with self.lock:
            # 将新文本添加到队列
            self.text_queue.append(str(text))
            self.total_pending_chars += len(str(text))

    def flush_remaining(self):
        """立即输出剩余所有内容"""
        with self.lock:
            # 输出当前正在处理的文本
            if self.current_text:
                sys.stdout.write(self.current_text)
                sys.stdout.flush()
                self.current_text = ""
            
            # 输出队列中的所有文本
            while self.text_queue:
                text = self.text_queue.popleft()
                sys.stdout.write(text)
                sys.stdout.flush()
            
            self.total_pending_chars = 0
            self.accumulated_chars = 0.0

    def _calculate_dynamic_speed(self, buffer_length: int) -> float:
        """计算动态输出速度"""
        if buffer_length <= 0:
            return self.min_speed
        
        # 使用组合函数计算速度
        exp_component = 1 - math.exp(-buffer_length / self.decay_factor)
        log_component = math.log(1 + buffer_length) / math.log(1 + self.decay_factor)
        combined_factor = 2 * exp_component * log_component / (exp_component + log_component + 1e-6)
        
        # 计算目标速度
        target_speed = self.min_speed + (self.max_speed - self.min_speed) * combined_factor
        
        # 平滑速度变化
        smooth_speed = (self.smoothing_factor * self.current_speed + 
                       (1 - self.smoothing_factor) * target_speed)
        
        self.current_speed = smooth_speed
        return smooth_speed

    def _output_processor(self):
        """输出处理线程"""
        last_update_time = time.time()
        
        while self.running or self.current_text or self.text_queue:
            try:
                current_time = time.time()
                time_delta = current_time - last_update_time
                last_update_time = current_time
                
                with self.lock:
                    # 如果当前没有正在处理的文本，从队列中取出下一个
                    if not self.current_text and self.text_queue:
                        self.current_text = self.text_queue.popleft()
                    
                    # 如果有文本需要输出
                    if self.current_text:
                        # 计算动态速度（基于总待打印字符数）
                        dynamic_speed = self._calculate_dynamic_speed(self.total_pending_chars)
                        
                        # 计算本次应该输出的字符数
                        chars_to_output = dynamic_speed * time_delta + self.accumulated_chars
                        actual_chars = int(chars_to_output)
                        self.accumulated_chars = chars_to_output - actual_chars
                        
                        # 输出字符
                        if actual_chars > 0:
                            chars_to_print = min(actual_chars, len(self.current_text))
                            to_print = self.current_text[:chars_to_print]
                            self.current_text = self.current_text[chars_to_print:]
                            
                            # 更新总待打印字符数
                            self.total_pending_chars = max(0, self.total_pending_chars - chars_to_print)
                            
                            # 输出到控制台
                            sys.stdout.write(to_print)
                            sys.stdout.flush()
                
                # 短暂休眠，控制更新频率
                time.sleep(0.02)  # 50Hz更新频率
                
            except Exception as e:
                # 错误处理：直接输出剩余内容
                if self.current_text:
                    sys.stdout.write(self.current_text)
                    sys.stdout.flush()
                    self.current_text = ""

    @property
    def buffer_size(self) -> int:
        """获取当前待打印的总字符数"""
        with self.lock:
            queue_chars = sum(len(text) for text in self.text_queue)
            return len(self.current_text) + queue_chars

    @property
    def is_running(self) -> bool:
        """检查系统是否正在运行"""
        return self.running

    @property
    def queue_length(self) -> int:
        """获取队列中的文本块数量"""
        with self.lock:
            return len(self.text_queue)

# 创建全局实例
_global_print_stream = PrintStream()

def print_stream(*args, sep: str = ' ', end: str = '\n', flush: bool = False) -> None:
    """
    动态速度打印函数
    
    Args:
        *args: 要打印的内容
        sep: 分隔符，默认为空格
        end: 结尾字符，默认为换行符
        flush: 是否立即刷新，默认为False
    """
    try:
        # 确保系统已启动
        if not _global_print_stream.is_running:
            _global_print_stream.start()
        
        # 组合输出内容
        text = sep.join(str(arg) for arg in args) + end
        
        if flush:
            # 立即输出
            sys.stdout.write(text)
            sys.stdout.flush()
        else:
            # 添加到队列
            _global_print_stream.add_to_buffer(text)
            
    except Exception as e:
        # 如果出错，回退到标准打印
        print(*args, sep=sep, end=end)

def start_print_stream() -> None:
    """手动启动打印流系统"""
    _global_print_stream.start()

def stop_print_stream() -> None:
    """停止打印流系统"""
    _global_print_stream.stop()

def flush_print_stream() -> None:
    """立即输出所有缓冲区内容"""
    _global_print_stream.flush_remaining()

def get_buffer_size() -> int:
    """获取当前缓冲区大小"""
    return _global_print_stream.buffer_size

def get_queue_length() -> int:
    """获取队列中的文本块数量"""
    return _global_print_stream.queue_length

def is_print_stream_running() -> bool:
    """检查打印流系统是否正在运行"""
    return _global_print_stream.is_running

def set_print_speed(min_speed: float = 5.0, max_speed: float = 50.0) -> None:
    """
    设置打印速度范围
    
    Args:
        min_speed: 最小打印速度（字符/秒）
        max_speed: 最大打印速度（字符/秒）
    """
    _global_print_stream.min_speed = max(1.0, min_speed)
    _global_print_stream.max_speed = max(_global_print_stream.min_speed, max_speed)

def configure_print_stream(min_speed: float = 5.0, max_speed: float = 50.0, 
                          decay_factor: float = 20.0, smoothing_factor: float = 0.8) -> None:
    """
    配置打印流系统参数
    
    Args:
        min_speed: 最小打印速度
        max_speed: 最大打印速度  
        decay_factor: 衰减因子
        smoothing_factor: 平滑因子
    """
    _global_print_stream.min_speed = max(1.0, min_speed)
    _global_print_stream.max_speed = max(_global_print_stream.min_speed, max_speed)
    _global_print_stream.decay_factor = max(1.0, decay_factor)
    _global_print_stream.smoothing_factor = max(0.1, min(0.99, smoothing_factor))

# 注册退出时的清理函数
def _cleanup():
    """程序退出时的清理函数"""
    try:
        _global_print_stream.flush_remaining()
        _global_print_stream.stop()
    except Exception:
        pass

atexit.register(_cleanup)

# 导出的所有函数和类
__all__ = [
    'print_stream',
    'start_print_stream', 
    'stop_print_stream',
    'flush_print_stream',
    'get_buffer_size',
    'get_queue_length',
    'is_print_stream_running',
    'set_print_speed',
    'configure_print_stream',
    'PrintStream'
]

# 测试代码
if __name__ == "__main__":
    # 配置打印流
    configure_print_stream(min_speed=5, max_speed=50)
    
    # 测试有序打印
    print_stream("第一段文本：Hello World!")
    time.sleep(0.5)
    print_stream("第二段文本：这是一个测试。")
    time.sleep(0.3)
    print_stream("第三段文本：确保文本按顺序输出，不会交叉打印。")
    
    # 测试大量文本
    print_stream("\n" + "="*50)
    for i in range(5):
        print_stream(f"批量文本 {i+1}: " + "这是一段较长的测试文本。" * 3)
    
    # 等待输出完成
    time.sleep(10)
    stop_print_stream()
