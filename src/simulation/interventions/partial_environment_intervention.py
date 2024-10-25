from typing import Tuple
from datetime import date, timedelta
import random
from collections import namedtuple
from src.simulation.interventions.intervention import Intervention, make_routine_change_events
from src.simulation.event import DayEvent, ChangeEnvInterventionStateEffect
from src.world.environments import CityCommunity, SchoolInterventionState
from src.world import World
from src.simulation.params import Params

def no_school_routine(person):
    params = Params.loader()["interventions_routines"]["school_closure"]
    return {'school': params["school"], 'household': params['household'], 'city_community': params['city_community'],
            'neighborhood_community': params['neighborhood_community']}

def school_isolation_intervention(person):
    params = Params.loader()["interventions_routines"]["school_isolation"]
    return {'school': params["school"], 'household': params['household'],
            'city_community': params['city_community'], 'neighborhood_community': params['neighborhood_community']}


AttributeAndPeriodData = namedtuple("AttributeAndPeriodData", ["attribute", "period_time"])

class PartialEnvironmentIntervention(Intervention):
    """
    a base class for partial interventions which take a certain proportion of environments of a certain type,
    and applies some routine change to all people in them.
    The environments of that name are randomly shaffled (once per simulation),
    and each such intervention either takes the a prefix or a suffix of that order.
    """
    def __init__(
        self,
        start_date: date,
        duration: timedelta,
        compliance: float,
        proportion_of_envs: float,
        city_name: str,
        env_name: str,
        age_segment: Tuple[int, int],
        key: str,
        take_first: bool,
        routine_generator,
        args=None,
        period_data: AttributeAndPeriodData = None,
        # This means 'the _intervention_state of unaffected environments'.
        # For schools, it is SchoolInterventionState.OPEN
        unaffected_intervention_state=None,
        # This means 'the _intervention_state of affected environments'.
        # For schools, it is either
        # SchoolInterventionState.CLOSED or SchoolInterventionState.ISOLATED
        affected_intervention_state=None
    ):
        """
        :param start_date: datetime.date start date
        :param duration: timedelta duration
        :param compliance: float between 0 to 1
        :param proportion_of_envs: The proportion of the environments that the intervention should be applied to.
        :param city_name: str name
        :param env_name: str name
        :param age_segment: tuple of (int, int) of age segment (inclusive)
        :param key: str key for routine change
        :param take_first: bool, states whether to take the prefix or a suffix
        :param routine_generator: func to generate routine change
        :param args: args for the routine_generator
        :param period_data: An AttributeAndPerioData object which describes the period of the intervention
        :param unaffected_intervention_state: The environment state that means the environment is not affected, like OPEN
        :param affected_intervention_state: The environment state that means the environment is affected, like CLOSED
        """
        super(PartialEnvironmentIntervention, self).__init__(compliance, start_date, duration)
        self.proporion_of_envs = proportion_of_envs
        self._city_name = city_name
        self._env_name = env_name
        self.age_segment = age_segment
        self._key = key
        self._take_first = take_first
        self._routine_generator = routine_generator
        self._unaffected_intervention_state = unaffected_intervention_state
        self._affected_intervention_state = affected_intervention_state
        self._args = args
        self.period_data = period_data
        assert (self._unaffected_intervention_state is None) == \
               (self._affected_intervention_state is None), \
            "Must have either both or none of the state changes"

    def _condition(self, env):
        """
        checks whether the given environment is relevent for this intervention
        :param env: Environment object
        :return: True if the environment population matched the age segment
        """
        min_inside = self.age_segment[0] <= env._age_segment[0] <= self.age_segment[1]
        max_inside = self.age_segment[0] <= env._age_segment[1] <= self.age_segment[1]
        if min_inside != max_inside:
            raise Exception(
                "Trying to apply an intervention with a partially overlapping "
                "age segment to one of its environments! The age segment of "
                "the intervention is {} and of the environment is {}".format(
                    self.age_segment, env._age_segment
                ))
        return min_inside

    def generate_events(self, world: World):
        """
        creates the events that make the intervention happen in the simulation.
        :param world: World object
        :return: list of new Event object, for the simulation to register
        """
        new_events = []
        if self._city_name == 'all':
            cities = world.get_all_city_communities()
        else:
            cities = [world.get_city_community(self._city_name)]
        envs_taken = []
        for city in cities:
            assert isinstance(city, CityCommunity)
            curr_envs = city.get_sorted_environments(self._env_name)
            if self._take_first:
                first_not_taken = int(len(curr_envs) * self.proporion_of_envs)
                curr_envs_taken = curr_envs[:first_not_taken]
            else:
                # This is done so x from top and 1-x from bottom will cover exactly everything
                first_taken = int(len(curr_envs) * (1 - self.proporion_of_envs))
                curr_envs_taken = curr_envs[first_taken:]
            curr_envs_taken = [env for env in curr_envs_taken if self._condition(env)]
            for env in curr_envs_taken:
                envs_taken.append(env)
        for env in envs_taken:
            if self._affected_intervention_state is not None:
                new_events.append(
                    DayEvent(
                        date=self.start_date,
                        effect=ChangeEnvInterventionStateEffect(
                            env=env,
                            old_state=self._unaffected_intervention_state,
                            new_state=self._affected_intervention_state
                        )
                    )
                )
                new_events.append(
                    DayEvent(
                        date=self.end_date,
                        effect=ChangeEnvInterventionStateEffect(
                            env=env,
                            old_state=self._affected_intervention_state,
                            new_state=self._unaffected_intervention_state
                        )
                    )
                )
            if self.period_data is None:
                for person in env._person_dict:
                    if random.random() < self.compliance:
                        curr_events = make_routine_change_events(
                            person,
                            self.start_date,
                            self.end_date,
                            self._key,
                            self._routine_generator,
                            self._args
                        )
                        for event in curr_events:
                            new_events.append(event)
            else:
                for person in env._person_dict:
                    if random.random() < self.compliance:
                        start_date = self.start_date
                        end_date = self.start_date + self.period_data.period_time * self.period_data.attribute(person)
                        finished = False
                        while not finished:
                            if end_date >= self.end_date:
                                end_date = self.end_date
                                finished = True
                            if end_date > start_date:
                                curr_events = make_routine_change_events(
                                    person,
                                    start_date,
                                    end_date,
                                    self._key,
                                    self._routine_generator,
                                    self._args
                                )
                                for event in curr_events:
                                    new_events.append(event)

                            start_date = end_date + self.period_data.period_time
                            end_date = start_date + self.period_data.period_time * (self.period_data.attribute.num_possibilities - 1)

        return new_events

class SchoolClosureIntervention(PartialEnvironmentIntervention):
    """
    Implementation of partial school closure, that only applies to some of the people in the city,
    and then to the rest during the next period of time. The partition is by the given environment, and is random.
    The period are determined by the AttributeAndPeriodData
    """
    def __init__(
        self,
        start_date: date,
        duration: timedelta,
        compliance: float,
        proportion_of_envs: float,
        city_name: str,
        age_segment: Tuple[int, int],
        period_data: AttributeAndPeriodData = None
    ):
        super(SchoolClosureIntervention, self).__init__(
            start_date=start_date,
            duration=duration,
            compliance=compliance,
            proportion_of_envs=proportion_of_envs,
            city_name=city_name,
            env_name="school",
            age_segment=age_segment,
            key='school_closure',
            take_first=True,
            routine_generator=no_school_routine,
            unaffected_intervention_state=SchoolInterventionState.OPEN,
            affected_intervention_state=SchoolInterventionState.CLOSED,
            period_data=period_data
        )

class SchoolIsolationIntervention(PartialEnvironmentIntervention):
    """
    Implementation of partial school isolation, that only applies to some of the people in the city,
    and then to the rest during the next period of time. The partition is by the given environment, and is random.
    The period are determined by the AttributeAndPeriodData
    """
    def __init__(
        self,
        start_date: date,
        duration: timedelta,
        compliance: float,
        proportion_of_envs: float,
        city_name: str,
        age_segment: Tuple[int, int],
        period_data: AttributeAndPeriodData = None
    ):
        super(SchoolIsolationIntervention, self).__init__(
            start_date=start_date,
            duration=duration,
            compliance=compliance,
            proportion_of_envs=proportion_of_envs,
            city_name=city_name,
            env_name="school",
            age_segment=age_segment,
            key='school_isolation',
            take_first=False,
            routine_generator=school_isolation_intervention,
            unaffected_intervention_state=SchoolInterventionState.OPEN,
            affected_intervention_state=SchoolInterventionState.ISOLATED,
            period_data=period_data
        )
