import typing

import pydantic


class FrontendSpec(pydantic.BaseModel):
    jobs_dashboard_url: typing.Optional[str]
