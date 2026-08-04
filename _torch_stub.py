"""完整的 torch stub —— 让不含 torch 的环境能 unpickle 含 torch tensor 的 pickle 流，
并能重新 pickle 存回（保持对象身份）。所有 torch tensor 退化为 _Stub 对象。"""
import sys, types, importlib.abc, importlib.machinery, copyreg


class _Stub:
    """torch 对象的占位替身。支持 pickle round-trip（重建为自身）。
    实例本身也可被调用（pickle 时 torch._utils._rebuild_tensor(...) 会被当作构造器）。"""
    def __reduce__(self):
        return (_rebuild_stub, ())
    def __call__(self, *a, **k):
        return _Stub()


def _rebuild_stub():
    return _Stub()


class _FakeModule(types.ModuleType):
    """任何属性访问 / 调用都返回 _Stub（可 pickle）。"""
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Stub()
    def __call__(self, *a, **k):
        return _Stub()


class TorchFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name == "torch" or name.startswith("torch."):
            return importlib.machinery.ModuleSpec(name, self)
        return None
    def create_module(self, spec):
        m = _FakeModule(spec.name)
        # 让它被识别为 package，从而 torch._utils 等子模块能被 finder 接管
        m.__path__ = []
        m.__package__ = spec.name
        return m
    def exec_module(self, module):
        pass


sys.meta_path.insert(0, TorchFinder())
