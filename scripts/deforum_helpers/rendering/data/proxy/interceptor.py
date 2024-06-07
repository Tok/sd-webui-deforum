from aspectlib import Aspect, weave

from scripts.deforum_helpers.args import RootArgs
from .root_data_proxy import RootDataProxyWrapper


class RootInterceptor(Aspect):
    def __call__(self, initial_info):
        print(f"Intercepted line: root.initial_info = {initial_info}")
        yield

@Aspect
def after_return(self, _, args, kwargs, value):
    root = kwargs.get('root')
    print("_________root: " + root)
    print("_________root_proxy: " + str(self.root_proxy))

    self.root_proxy.initial_info = root.initial_info
    self.root_proxy.first_frame = root.first_frame

class UpdateRootProxyInterceptor(Aspect):
    root_proxy: RootDataProxyWrapper

    #def __init__(self, root_proxy):
    #    super().__init__(self, None)
    #    self.root_proxy = root_proxy

    def after_return(self, _, args, kwargs, value):
        root = kwargs.get('root')
        print("_________root: " + root)
        print("_________root_proxy: " + str(self.root_proxy))

        self.root_proxy.initial_info = root.initial_info
        self.root_proxy.first_frame = root.first_frame

        return value
