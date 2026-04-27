import logging
import sys
import inspect
import functools
from contextvars import ContextVar
from functools import wraps

_current_level = ContextVar("current_level", default=0)

# --- 配置部分保持不变 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout,
)


class HierLogger(logging.Logger):
    indent_str = "  "
    
    def info(self, msg, *args, **kwargs):
        lvl = _current_level.get()
        # 保持缩进逻辑
        indented_msg = f"{self.indent_str * lvl}{msg}"
        
        # 关键修复：
        # 如果 args 里面有内容，说明用户用了 logger.info("msg %s", arg)
        # 如果 args 为空，说明用户可能用了 f-string，直接传 indented_msg 即可
        if not args:
            super().info(indented_msg, **kwargs)
        else:
            super().info(indented_msg, *args, **kwargs)

logging.setLoggerClass(HierLogger)
logger = logging.getLogger("myhier")


# --- 工具函数 ---

def _get_func_name(func):
    """安全获取函数或可调用对象的名称"""
    # 优先获取 qualname (带类名前缀)，其次 name，最后转字符串
    return getattr(func, '__qualname__', getattr(func, '__name__', str(func)))


def enter():
    lvl = _current_level.get()
    _current_level.set(lvl + 1)


def exit():
    lvl = _current_level.get()
    _current_level.set(max(0, lvl - 1))


# --- 核心装饰器改进 ---

def logged(func):
    """通用函数装饰器：支持层级缩进和安全名称访问"""
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 动态获取 logger，如果函数定义在类里，使用模块名
        func_logger = logging.getLogger(func.__module__)
        func_name = _get_func_name(func)
        
        func_logger.info(f"▶ entering {func_name}")
        enter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            exit()
            func_logger.info(f"◀ leaving {func_name}")
    
    return wrapper


def logged_class(cls):
    """
    类装饰器：为类中所有方法添加具有层级感的日志。
    改进：处理了方法包装逻辑，确保其与层次结构兼容。
    """
    # 遍历类的所有属性，而不仅仅是函数
    for name in dir(cls):
        # 忽略私有方法（__call__ 除外）
        if name.startswith("_") and name != "__call__":
            continue
        
        value = getattr(cls, name)
        
        # 只装饰可调用对象
        if not callable(value):
            continue
        
        # 如果已经是被 logged 装饰过的，避免重复装饰
        if hasattr(value, "__wrapped_by_hier__"):
            continue
        
        # 判断是否为静态方法或类方法需要特殊处理，这里以普通方法为主
        # 重新包装方法
        @functools.wraps(value)
        def make_wrapper(func=value):
            @logged
            @functools.wraps(func)
            def wrapped(*args, **kwargs):
                return func(*args, **kwargs)
            
            # 标记已装饰
            wrapped.__wrapped_by_hier__ = True
            return wrapped
        
        setattr(cls, name, make_wrapper(value))
    
    return cls


# 6. 默认导出一个实例供直接使用
logging.setLoggerClass(HierLogger)
logger = logging.getLogger("myhier")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
logger.addHandler(handler)

# 暴露接口
__all__ = ["logger", "logged", "logged_class", "enter", "exit"]
