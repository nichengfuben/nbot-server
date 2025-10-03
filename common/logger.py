import re
import sys
import traceback
from typing import Optional, Union

from printstream import print_stream


_LEADING_TAG_PATTERN = re.compile(r"^\[[^\]]+\]")


class Logger:
    def __init__(self, name: Optional[str] = None, front_mounted: bool = True):
        self.name = (name or '').strip()
        self.front_mounted = front_mounted

    def _format(self, message: str) -> str:
        if not message:
            return ''
        
        # 如果 front_mounted 为 False，不添加 [TAG]
        if not self.front_mounted:
            return message
            
        # 如果消息已经以 [TAG] 开头，则不再重复添加前缀，保证控制台输出保持不变
        if _LEADING_TAG_PATTERN.match(message):
            return message
        if self.name:
            return f"[{self.name.upper()}] {message}"
        return message

    def _get_exc_info(self, exc_info: Union[bool, tuple]) -> str:
        """
        获取异常信息
        
        Args:
            exc_info: 
                - True: 使用 sys.exc_info() 获取当前异常
                - tuple: 直接使用提供的 (exc_type, exc_value, exc_tb)
                - False: 不获取异常信息
        
        Returns:
            格式化的异常堆栈字符串
        """
        if not exc_info:
            return ""
        
        if exc_info is True:
            exc_type, exc_value, exc_tb = sys.exc_info()
        elif isinstance(exc_info, tuple) and len(exc_info) == 3:
            exc_type, exc_value, exc_tb = exc_info
        else:
            return ""
        
        if exc_type is None:
            return ""
        
        return ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))

    def _log(
        self, 
        message: str, 
        end: str = "\n", 
        flush: bool = False, 
        exc_info: Union[bool, tuple] = False
    ) -> None:
        """
        内部统一的日志输出方法
        
        Args:
            message: 日志消息
            end: 行结束符
            flush: 是否立即刷新输出缓冲区
            exc_info: 异常信息，可以是 True、False 或异常元组
        """
        msg = str(message)
        
        # 处理异常信息
        if exc_info:
            exc_text = self._get_exc_info(exc_info)
            if exc_text:
                msg = f"{msg}\n{exc_text}" if msg else exc_text
        
        formatted = self._format(msg)
        
        # 当调用者需要自定义 end/flush 时，使用内置 print 以确保行为与原先一致
        if end != "\n" or flush:
            print(formatted, end=end, flush=flush)
        else:
            print_stream(formatted)

    def info(
        self, 
        message: str, 
        end: str = "\n", 
        flush: bool = False, 
        exc_info: Union[bool, tuple] = False
    ) -> None:
        """
        输出 INFO 级别日志
        
        Args:
            message: 日志消息
            end: 行结束符，默认换行
            flush: 是否立即刷新输出缓冲区
            exc_info: 是否包含异常信息，True 表示自动获取当前异常，也可传入异常元组
        """
        self._log(message, end=end, flush=flush, exc_info=exc_info)

    def debug(
        self, 
        message: str, 
        end: str = "\n", 
        flush: bool = False, 
        exc_info: Union[bool, tuple] = False
    ) -> None:
        """
        输出 DEBUG 级别日志
        
        Args:
            message: 日志消息
            end: 行结束符，默认换行
            flush: 是否立即刷新输出缓冲区
            exc_info: 是否包含异常信息，True 表示自动获取当前异常，也可传入异常元组
        """
        self._log(message, end=end, flush=flush, exc_info=exc_info)

    def warning(
        self, 
        message: str, 
        end: str = "\n", 
        flush: bool = False, 
        exc_info: Union[bool, tuple] = False
    ) -> None:
        """
        输出 WARNING 级别日志
        
        Args:
            message: 日志消息
            end: 行结束符，默认换行
            flush: 是否立即刷新输出缓冲区
            exc_info: 是否包含异常信息，True 表示自动获取当前异常，也可传入异常元组
        """
        self._log(message, end=end, flush=flush, exc_info=exc_info)

    def error(
        self, 
        message: str, 
        end: str = "\n", 
        flush: bool = False, 
        exc_info: Union[bool, tuple] = False
    ) -> None:
        """
        输出 ERROR 级别日志
        
        Args:
            message: 日志消息
            end: 行结束符，默认换行
            flush: 是否立即刷新输出缓冲区
            exc_info: 是否包含异常信息，True 表示自动获取当前异常，也可传入异常元组
        """
        self._log(message, end=end, flush=flush, exc_info=exc_info)

    def critical(
        self, 
        message: str, 
        end: str = "\n", 
        flush: bool = False, 
        exc_info: Union[bool, tuple] = False
    ) -> None:
        """
        输出 CRITICAL 级别日志
        
        Args:
            message: 日志消息
            end: 行结束符，默认换行
            flush: 是否立即刷新输出缓冲区
            exc_info: 是否包含异常信息，True 表示自动获取当前异常，也可传入异常元组
        """
        self._log(message, end=end, flush=flush, exc_info=exc_info)

    def exception(
        self, 
        message: str, 
        end: str = "\n", 
        flush: bool = False, 
        exc_info: Union[bool, tuple] = True
    ) -> None:
        """
        输出异常日志，默认自动包含异常信息
        
        Args:
            message: 日志消息
            end: 行结束符，默认换行
            flush: 是否立即刷新输出缓冲区
            exc_info: 异常信息，默认为 True 自动获取当前异常
        """
        self._log(message, end=end, flush=flush, exc_info=exc_info)

    def log(
        self, 
        message: str, 
        end: str = "\n", 
        flush: bool = False, 
        exc_info: Union[bool, tuple] = False
    ) -> None:
        """
        通用日志输出方法
        
        Args:
            message: 日志消息
            end: 行结束符，默认换行
            flush: 是否立即刷新输出缓冲区
            exc_info: 是否包含异常信息，True 表示自动获取当前异常，也可传入异常元组
        """
        self._log(message, end=end, flush=flush, exc_info=exc_info)

    # 别名方法
    warn = warning  # warning 的别名


def get_logger(name: Optional[str] = None, front_mounted: bool = True) -> Logger:
    """
    获取 Logger 实例
    
    Args:
        name: Logger 名称，会被转换为大写作为标签
        front_mounted: 是否在消息前添加 [TAG] 标签，默认 True
    
    Returns:
        Logger 实例
    """
    return Logger(name, front_mounted)


__all__ = [
    'Logger',
    'get_logger',
]
