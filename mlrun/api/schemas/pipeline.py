import typing

import pydantic


class PipelinesPagination(str):
    default_page_size = 20
    # https://github.com/kubeflow/pipelines/blob/master/backend/src/apiserver/list/list.go#L363
    max_page_size = 200


class PipelinesOutput(pydantic.BaseModel):
    # use the format query param to control what is returned
    runs: typing.List[typing.Union[dict, str]]
    total_size: int
    next_page_token: typing.Optional[str]
