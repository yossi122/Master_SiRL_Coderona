from src.world.environments.homogeneous_environment import HomogeneousEnvironment
from enum import Enum

class SchoolInterventionState(Enum):
    __slots__ = ()
    OPEN = 1      # Students go in and out
    CLOSED = 2    # Students stay out
    ISOLATED = 3  # Students stay in


class School(HomogeneousEnvironment):
    __slots__ = ('_city', '_age_segment', '_city_env', '_intervention_state')
    name = "school"

    def __init__(self, city, contact_prob_between_each_two_people, age_segment, full_name=None):
        super(School, self).__init__(contact_prob_between_each_two_people, full_name=full_name)
        self._city = city
        self._city_env = None
        self._age_segment = age_segment
        self._intervention_state = SchoolInterventionState.OPEN

    def set_city_env(self, city_env):
        self._city_env = city_env

    def set_intervention_state(self, old_state, new_state):
        if self._intervention_state == old_state:  # !@
            assert self._intervention_state == old_state, (
                "Trying to switch a school from state {} to state {}, "
                "but that school is in state {}!\n"
                "This probably happened since SchoolIsolation and SchoolClosure "
                "add up to more than 100% in the same age_group X city X time."
            ).format(old_state, new_state, self._intervention_state)
            self._intervention_state = new_state
