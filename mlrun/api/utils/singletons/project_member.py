import mlrun.api.utils.projects.follower
import mlrun.api.utils.projects.leader
import mlrun.api.utils.projects.member
import mlrun.config

# TODO: something nicer
project_member: mlrun.api.utils.projects.member.Member = None


def initialize_project_member():
    global project_member
    if mlrun.config.config.httpdb.projects.leader in ["mlrun", "nop-self-leader"]:
        project_member = mlrun.api.utils.projects.leader.Member()
        project_member.initialize()
    else:
        project_member = mlrun.api.utils.projects.follower.Member()
        project_member.initialize()


def get_project_member() -> mlrun.api.utils.projects.member.Member:
    global project_member
    return project_member
