from src.world.environments.environment import Environment


class InitialGroup(Environment):
    name = "initial_group"
    singleton = None

    def __init__(self):
        super(InitialGroup, self).__init__("initial_group")

    @classmethod
    def initial_group(cls):
        if not cls.singleton:
            cls.singleton = InitialGroup()
        return cls.singleton
