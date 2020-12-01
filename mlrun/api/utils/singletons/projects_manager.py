import mlrun.api.utils.projects.members.leader

# TODO: something nicer
projects_member: mlrun.api.utils.projects.members.leader.Member = None


def initialize_projects_manager():
    global projects_member
    # currently we're always leaders, when there will be follower member implementation, we should condition which one
    # to initialize here
    projects_member = mlrun.api.utils.projects.members.leader.Member()
    projects_member.initialize()


def get_projects_member() -> mlrun.api.utils.projects.members.leader.Member:
    global projects_member
    return projects_member
