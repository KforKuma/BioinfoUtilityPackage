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
    """
    Mark a function, method, or class as deprecated.

    Parameters
    ----------
    since : str, optional
        Version in which the object was deprecated.
    removal : str, optional
        Version in which the object is expected to be removed.
    alternative : str, optional
        Suggested alternative API.
    message : str, optional
        Custom deprecation message (overrides auto-generated message).
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
        
        # 类也能工作（__init__ 被 wrapper 包住）
        return wrapper
    
    return decorator
