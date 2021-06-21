import fastapi

import mlrun.api.utils.authorizers.authorizer


class Authorizer(mlrun.api.utils.authorizers.authorizer.Authorizer):
    def authorize(self, request: fastapi.Request):
        pass
