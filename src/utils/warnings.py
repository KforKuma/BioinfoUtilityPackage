from __future__ import annotations
import functools
import warnings


def deprecated(
        *,
        since: str | None = None,
        removal: str | None = None,
        alternative: str | None = None,
        message: str | None = None,
):
    """标记函数、方法或类已废弃。

    Args:
        since: 开始废弃的版本。
        removal: 预计移除版本。
        alternative: 推荐替代 API。
        message: 自定义 warning 文本；提供时会覆盖自动生成文本。

    Returns:
        装饰器函数。

    Example:
        >>> @deprecated(since="0.2", alternative="new_func")
        ... def old_func():
        ...     return 1
        >>> old_func()
        1
        # 调用时会触发 DeprecationWarning。
    """
    
    def decorator(obj):
        obj_name = obj.__qualname__
        
        parts = [f"`{obj_name}` is deprecated"]
        
        if since:
            parts.append(f"since {since}")
        if removal:
            parts.append(f"and will be removed in {removal}")
        
        if alternative:
            parts.append(f"; use `{alternative}` instead")
        
        default_msg = " ".join(parts) + "."
        
        warn_msg = message or default_msg
        
        @functools.wraps(obj)
        def wrapper(*args, **kwargs):
            warnings.warn(
                warn_msg,
                category=DeprecationWarning,
                stacklevel=2,
            )
            return obj(*args, **kwargs)
        
        # 返回 wrapper 而不改对象主体，便于兼容函数、方法和简单类调用。
        return wrapper
    
    return decorator
