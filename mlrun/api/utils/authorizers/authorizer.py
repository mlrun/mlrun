import abc

import fastapi


class Authorizer(abc.ABC):
    @abc.abstractmethod
    def authorize(self, request: fastapi.Request):
        pass
