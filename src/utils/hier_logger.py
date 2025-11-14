import logging,sys
from contextvars import ContextVar
from functools import wraps

_current_level = ContextVar("current_level", default=0)
# -------------------
# 配置 root logger
# -------------------
logging.basicConfig(
    level=logging.INFO,               # INFO 及以上会显示
    format="%(message)s",             # 只显示消息（hierarchical decorator 会加缩进）
    stream=sys.stdout,                # 输出到控制台
)
def get_logger(name):
    return logging.getLogger(name)

class HierLogger(logging.Logger):
    indent_str = "  "

    def info(self, msg, *args, **kwargs):
        lvl = _current_level.get()
        indented_msg = f"{self.indent_str * lvl}{msg}"
        super().info(indented_msg, *args, **kwargs)

def enter():
    lvl = _current_level.get()
    _current_level.set(lvl + 1)

def exit():
    lvl = _current_level.get()
    _current_level.set(max(0, lvl - 1))

def logged(func):
    """Decorator: before call -> indent +1, after call -> indent -1."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        enter()
        try:
            return func(*args, **kwargs)
        finally:
            exit()
    return wrapper

# 注册 logger 类
logging.setLoggerClass(HierLogger)
logger = logging.getLogger("myhier")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
logger.addHandler(handler)

__all__ = ["logger", "logged", "enter", "exit"]
