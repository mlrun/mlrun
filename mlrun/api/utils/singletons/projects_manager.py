from mlrun.api.utils.projects.manager import ProjectsManager

# TODO: something nicer
projects_manager: ProjectsManager = None


def initialize_projects_manager():
    global projects_manager
    projects_manager = ProjectsManager()
    projects_manager.start()


def get_projects_manager() -> ProjectsManager:
    global projects_manager
    return projects_manager
