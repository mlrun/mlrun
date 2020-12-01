import mlrun.api.utils.projects.members.leader

# TODO: something nicer
project_member: mlrun.api.utils.projects.members.leader.Member = None


def initialize_project_member():
    global project_member
    # currently we're always leaders, when there will be follower member implementation, we should condition which one
    # to initialize here
    project_member = mlrun.api.utils.projects.members.leader.Member()
    project_member.initialize()


def get_project_member() -> mlrun.api.utils.projects.members.leader.Member:
    global project_member
    return project_member
