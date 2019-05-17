import uuid
import getpass

from mlrun.execution import KFPClientCtx


class LocalRuntime(KFPClientCtx):

    def __init__(self, name, parameters={}):
        uid = uuid.uuid1().hex
        KFPClientCtx.__init__(self, uid)
        self.name = name
        self.parent_type = 'local'
        self.owner = getpass.getuser()
        self._parameters = parameters


