from .default import *

# 如果有个性化设置
try:
    from .local import *
except ImportError:
    pass