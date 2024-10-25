from collections import namedtuple
from copy import copy
from datetime import date, timedelta
from enum import Enum
import numpy as _np

from src.simulation.event import (
    Event,
    DayEvent,
    EmptyTrigger,
    DiseaseStateChangeEffect
)
from src.seir import DiseaseState, sample_seir_times
from src.simulation.params import Params
from src.util.Enumerations import machine_type
from src.world.infection_data import InfectionData

RedactedPerson = namedtuple("RedactedPerson", ("age", "disease_state"))
RedactedPersonAndEnv = namedtuple("RedactedPersonAndEnv", ("age", "disease_state", "infection_env_source"))


class Person(object):
    """
    A person in the simulation.
    A person has a daily routine, that states what are the environments he visits on the next day.
    Also, he has disease state that indicates wht stage of the disease he's at.
    See the design document for more details
    """
    __slots__ = (
        '_changed',
        '_age',
        '_environments',
        '_current_routine',
        '_infectiousness_prob',
        '_disease_state',
        'is_susceptible',
        'is_dead',
        'is_infectious',
        'is_infected',
        'is_vaccinated',
        '_id',
        'state_to_events',
        'routine_change_multiplicities',
        'routine_changes',
        '_infection_data',
        '_num_infections',
        'last_state',
        '_seir_times',
        'state_machine_type',
        '_minimum_infectiousness_age',
        '_my_neighborhood',
    )
    num_people_so_far = 0

    def __init__(self, age, environments=None):
        params = Params.loader()['population']
        R0 = params["R0_percent"]
        StartAsRecovered = False
        #if random.random() < R0:
        #   StartAsRecovered = True

        self._changed = True
        if not environments:
            environments = []
        self._age = age
        assert len(set([env.name for env in environments])) == len(environments), "Got duplicate environment names"
        self._environments = {env.name: env for env in environments}
        self._current_routine = {env_name: 1 for env_name in self._environments}
        params = Params.loader()['person']
        self._minimum_infectiousness_age = params['minimum_infectiousness_age']
        if age <= self._minimum_infectiousness_age:
            self._infectiousness_prob = 0.0
        else:
            self._infectiousness_prob = \
                min(params['base_infectiousness'] * \
                    _np.random.gamma(
                        params['individual_infectiousness_gamma_shape'],
                        params['individual_infectiousness_gamma_scale']
                    ), 1)
        # if StartAsRecovered:
        #   self._disease_state = DiseaseState.IMMUNE
        #   self.is_susceptible = False
        #   self.is_infected = True
        # else:
        self._disease_state = DiseaseState.SUSCEPTIBLE
        self.is_susceptible = True
        self.is_dead = False
        self.is_infectious = False
        self.is_infected = False
        str_type = params['state_machine_type']
        assert str_type in ['SIRS', 'SIR']
        self.state_machine_type = machine_type[str_type]
        self._id = Person.num_people_so_far
        # hold all the events that are triggered by some disease state(s) change(s), like isolation when symptomatic
        self.state_to_events = {}
        # The following counts the number of different interventions that force each routine change on this person.
        # For instance, I might be in quarantine because I'm old and because I'm symptomatic.
        # Without this counter, people could go into quarantine because they're old, get symptoms during quarantine,
        # then go out of quarantine when the symptoms pass.
        self.routine_change_multiplicities = {}
        self.routine_changes = {}
        self._infection_data = None
        # Table that currespond to seir times and events so it will be easier to mange
        self._seir_times = None
        self._my_neighborhood = None
        self._num_infections = 0
        #if StartAsRecovered:
        #    self.last_state =RedactedPerson(self.get_age(), self.get_disease_state())
        #else:
        self.last_state = None
        Person.num_people_so_far += 1

    def _init_event(self, old_state, new_state):
        """
        Init the state_to_events dict at the key (old_state, new_state), so we can hook other events here.
        :param old_state: the old disease state the person had
        :param new_state: the new disease state the person is changing to
        """
        states = old_state, new_state
        if states in self.state_to_events:
            return
        self.state_to_events[states] = Event(
            trigger=EmptyTrigger(),
            effect=DiseaseStateChangeEffect(
                person=self,
                old_state=old_state,
                new_state=new_state
            )
        )

    def make_eventless_copy(self):
        """
        make a copy of this Person object, but without any events or routine changes.
        This copy is for the disease data mainly.
        :return: new_person
        """
        new_person = copy(self)
        new_person.state_to_events = {}
        new_person.routine_change_multiplicities = {}
        new_person.routine_changes = {}
        return new_person

    def hook_on_change(self, states, event):
        """
        hook the given event on the given states change
        :param states: the state change to trigger the event
        :param event: the event to apply when the state change happen
        """
        self._init_event(*states)
        self.state_to_events[states].hook(event)

    def add_environment(self, environment):
        """
        Adds a new environment to the person, and also to he's routine.
        This should only be called when generating a population! not mid-simulation!
        """
        assert environment.name not in self._current_routine
        if environment.name == "neighborhood_community":
            self._my_neighborhood = environment
        self._environments[environment.name] = environment
        self._current_routine[environment.name] = 1
        self._change()

    def get_neighberhood(self):
        return self._my_neighborhood

    def get_age(self):
        """
        get the person's age
        """
        return self._age

    def get_routine(self):
        """
        get person's routine
        """
        return self._current_routine

    def get_environment(self, name):
        """
        get environment by name
        :param name: string that states the environment name
        :return: environment object
        """
        assert name in self._environments, "Unknown environment: '%s'" % name
        return self._environments[name]

    def has_environment(self, name):
        """
        return True if the person has an environment of that name
        :param name: string - the environment name
        :return: bool
        """
        return name in self._environments

    def register_to_daily_environments(self):
        """
        Sign up for all of the person's environments according to the weights in his routine.
        This function triggers the iterations (and therefore the infections) in the environments.
        """
        if not self._changed:
            return
        for env_name, env in self._environments.items():
            env.sign_up_for_today(self, self._current_routine[env_name])

    def _change(self):
        """
        update person's state due to change in his disease state
        We're tracking when a person is "changed" to save time.
        """
        self.is_susceptible = self._disease_state.is_susceptible()
        self.is_infectious = self._disease_state.is_infectious()
        self.is_infected = self._disease_state.is_infected()
        self.is_dead = self._disease_state.is_dead()
        self.is_vaccinated = self._disease_state.is_vaccinated()
        # print("in _change id:{} is_susceptible:{}".format(self.get_id(),self.is_susceptible))
        self._changed = True

    def get_prob_to_infect_on_contact(self):
        """
        return the probability that this person will infect on contact with other person.
        :return: int
        """
        return self._infectiousness_prob * self.get_disease_state().get_infectiousness_factor()

    def get_disease_state(self):
        """
        returns the person current disease state
        :return: DiseaseState
        """
        return self._disease_state

    def get_age_category(self):
        """
        :return: returns the person's age group, i.e 0, 10, 20...
        """
        return self._age - self._age % 10

    def get_infection_data(self):
        """
        gets the person's infection data, that states were and by whom he got infected.
        :return: InfectionData object
        """
        return self._infection_data

    def get_id(self):
        """
        get the person's unique id in the population
        :return: int
        """
        return self._id

    def get_state(self):
        """
        Returns the person age and disease state
        :return: tuple of int and diseaseState
        """
        return RedactedPerson(self.get_age(), self.get_disease_state())

    def save_state(self):
        """
        after handling all changed people, we save their states as the last state and reset the changed bool.
        """
        self.last_state = self.get_state()
        self._changed = False

    def get_last_state(self):
        """
        returns the person last state, before current one.
        """
        return self.last_state

    def set_disease_state(self, new_disease_state):
        """
        set new disease state, update all the disease attributes on the person object.
        this action makes the person to be considered "changed"
        :param new_disease_state: DiseaseState object
        """
        assert isinstance(new_disease_state, DiseaseState)
        self._disease_state = new_disease_state
        # print("in set_disease_state id:{} new_disease_state:{}".format(self.get_id(),new_disease_state))
        self._change()
        if new_disease_state.is_static():
            self._clear()

    def _clear(self):
        """
        clears all person's events
        """
        self.state_to_events = {}

    def gen_and_register_events_from_seir_times(self, date, states_and_times):
        """
        Create all the disease course (timeline) of this person -
        makes day events to each disease state, as the person goes from being infected,
        all the way to immunity/death (being removed)
        :param date: current date to start the seir times from
        :param states_and_times: disease states and their duration
        :return: all new events to add, that change the person's disease state
        """
        events = []
        last_state = self._disease_state
        curr_date = date
        for i in range(1, len(states_and_times)):
            # Only bind the events to time
            curr_date += states_and_times[i - 1][1]
            old_state = last_state
            new_state = states_and_times[i][0]
            new_event = DayEvent(curr_date)
            self._init_event(old_state, new_state)
            new_event.hook(self.state_to_events[(old_state, new_state)])
            events.append(new_event)
            last_state = states_and_times[i][0]
        assert states_and_times[-1][1] is None
        return events

    def infect_and_get_events(
            self,
            date,
            environment,
            infection_transmitter=None,
            seir_times=None
    ):
        """
        Infect this person, save his infection data, create and register the events of his state changes.
        :param date: data of infection data
        :param environment: where the person got infected
        :param infection_transmitter: the person who infected this person
        :param seir_times: The state changes of this person (what they are and how long they last).
        If this is None, it is sampled with the distribution defined in params.json.
        :return: infection events
        """

        assert self._disease_state == DiseaseState.SUSCEPTIBLE, "person state:" + str(self._disease_state)
        if self._age <= self._minimum_infectiousness_age:
            return []

        if seir_times:
            states_and_times = seir_times
        elif self._seir_times:
            states_and_times = self._seir_times
        else:
            states_and_times = sample_seir_times(self.state_machine_type, self)
        #update self._infection_data
        for state, date_change in states_and_times:
            if state != DiseaseState.SUSCEPTIBLE:
                break
        assert ((self.state_machine_type == machine_type.SIR) and (self._infection_data is None)) or \
               (self.state_machine_type == machine_type.SIRS), \
            "Infecting someone who is already infected id:{} machine_type:{}".format(str(self.get_id()),
                                                                                     self.state_machine_type.name)
        self.set_disease_state(DiseaseState.LATENT)
        self._infection_data = InfectionData(self, date + date_change, environment, infection_transmitter)
        # print("in infect_and_get_events id:{} disease_state:{} states_and_times:{}"\
        #     .format(self.get_id(),self.get_disease_state(),states_and_times))
        #update seir times table
        self._seir_times = states_and_times
        return self.gen_and_register_events_from_seir_times(date, states_and_times)

    def immune_and_get_events(
            self,
            start_date,
            delta_time,
            seir_times=None,
            immune_type=None
    ):
        """
        Immune this person, create and register the events of his state changes.
        :param start_date: date_from which we count the delta_time should be init to INIAL_DATE
        :param delta_time: time_delta from start_Date of the simulation in which the person will be immuned
        :param seir_times: The state changes of this person (what they are and how long they last).
        If this is None, it is sampled with the distribution defined in params.json.
        :return: tuple  first element true if person can be vaccinate
                        second elemnt is infection events 
        """
        lastState = self._disease_state
        if self._disease_state not in (DiseaseState.IMMUNE, DiseaseState.VACCINATED):  #TODO: just do nothing with already immuned?
            #print('immune_and_get_events:', self._disease_state)
            # !@ added DiseaseState.SYMPTOMATICINFECTIOUS+
            assert (self._disease_state in (DiseaseState.SUSCEPTIBLE,
                                            DiseaseState.LATENT,
                                            DiseaseState.ASYMPTOMATICINFECTIOUS,
                                            DiseaseState.INCUBATINGPOSTLATENT,
                                            DiseaseState.SYMPTOMATICINFECTIOUS
                                            )), \
                f"expected SUSCEPTIBLE or LATENT by found {str(self._disease_state)}"
            self.set_disease_state(DiseaseState.LATENT)
            if seir_times:
                states_and_times = seir_times
            elif self._seir_times:
                states_and_times = self._seir_times
            else:
                states_and_times = sample_seir_times(self.state_machine_type, self)
            if len(states_and_times) == 1:
                print(states_and_times)
                print('@#$%$#^$%*&)U')
                return False, []

            if len(states_and_times) > 1:
                #alt_state (alternative state) in case the person won't be immuned
                alt_state = states_and_times[-1]
                states_and_times = states_and_times[:-1]
                #Orgenize the states_and_times dictionaery so that is simulation.current_date == date this person will be immmune
                # only if he should get immuned
                i = 0
                new_states_and_times = []
                cut_in_middle = False
                #print('\t>>', len(states_and_times),states_and_times)
                zero_days = (delta_time.days == 0) and (
                            states_and_times[0][0] in [DiseaseState.SUSCEPTIBLE, DiseaseState.LATENT])
                #specify if the person can be vaccinated or we should find a substitute
                ok = True
                append_rest_of_table = False
                if zero_days:
                    if immune_type == 'vaccine':
                        new_states_and_times.append((DiseaseState.VACCINATED, timedelta(days=0)))
                        new_states_and_times.append((DiseaseState.VACCINATED, None))
                        lastState = DiseaseState.VACCINATED
                    else:
                        new_states_and_times.append((DiseaseState.IMMUNE, timedelta(days=0)))
                        new_states_and_times.append((DiseaseState.IMMUNE, None))
                        lastState = DiseaseState.IMMUNE
                    ok = True
                else:
                    while i < len(states_and_times):
                        if delta_time.days >= states_and_times[i][1].days:
                            new_states_and_times.append( \
                                (states_and_times[i][0], timedelta(days=states_and_times[i][1].days)))
                            delta_time -= timedelta(days=states_and_times[i][1].days)
                            lastState = states_and_times[i][0]
                        i += 1
                    if delta_time.days > 0:
                        if lastState in [DiseaseState.SUSCEPTIBLE, DiseaseState.LATENT]:
                            if immune_type == 'vaccine':
                                new_states_and_times.append((DiseaseState.VACCINATED, timedelta(delta_time.days)))
                            else:
                                new_states_and_times.append((DiseaseState.IMMUNE, timedelta(delta_time.days)))
                            ok = True
                        else:
                            append_rest_of_table = True
                            ok = False
                        delta_time -= timedelta(delta_time.days)
                    elif delta_time.days == 0:
                        if lastState in [DiseaseState.SUSCEPTIBLE, DiseaseState.LATENT]:
                            if immune_type == 'vaccine':
                                new_states_and_times.append((DiseaseState.IMMUNE, timedelta(delta_time.days)))
                            else:
                                new_states_and_times.append((DiseaseState.VACCINATED, timedelta(delta_time.days)))
                            ok = True
                        elif len(new_states_and_times) < len(states_and_times):
                            i -= 1
                            append_rest_of_table = True
                            ok = False
                    elif delta_time.days < 0:
                        cut_in_middle = True
                        if lastState in [DiseaseState.SUSCEPTIBLE, DiseaseState.LATENT]:
                            if immune_type == 'vaccine':
                                new_states_and_times[i - 1][0] == DiseaseState.VACCINATED
                                lastState = DiseaseState.VACCINATED
                            else:
                                new_states_and_times[i - 1][0] == DiseaseState.IMMUNE
                                lastState = DiseaseState.IMMUNE
                            ok = True
                        else:
                            #add the rest of the table don't change a thing
                            append_rest_of_table = True
                            ok = False

                    assert delta_time.days >= 0, 'miscalculating days'
                    # if delta_time.days < 0 :
                    #     print("i:{} len(states_and_times):{} delta_time.days:{}".format(i,len(states_and_times),delta_time.days))
                    #     print("start_date:{} ,old_delta_time:{} ,delta_time:{}".format(start_date,old_delta_time , delta_time))
                    #     print(states_and_times)
                    #     print("-----------------------")
                    if append_rest_of_table:
                        while i < len(states_and_times):
                            new_states_and_times.append((states_and_times[i][0], timedelta(days=states_and_times[i][1].days)))
                            i += 1

                #print("immune_and_get_events id:{} new_states_and_times:{}".format(self.get_id(),new_states_and_times))
                #At the End of the table we put immune only if the person was susptible or latent
                # otherwise he continues his life regularly
                if (not (zero_days)):
                    if (lastState in [DiseaseState.SUSCEPTIBLE, DiseaseState.LATENT]):
                        if immune_type == 'vaccine':
                            new_states_and_times.append((DiseaseState.VACCINATED, None))
                        else:
                            new_states_and_times.append((DiseaseState.IMMUNE, None))
                        ok = True
                    else:
                        new_states_and_times.append(alt_state)
                        ok = False
                # print(new_states_and_times)
                #Update person seir_times
                self._seir_times = new_states_and_times
                return ok, self.gen_and_register_events_from_seir_times(start_date, new_states_and_times)

    def add_routine_change(self, key, value):
        #print('add_routine_change',key, value)
        if key in self.routine_changes:
            assert key in self.routine_change_multiplicities
            self.routine_change_multiplicities[key] += 1
            assert self.routine_changes[key] == value
        else:
            assert key not in self.routine_change_multiplicities
            self.routine_changes[key] = value
            self.routine_change_multiplicities[key] = 1
            self.update_routine()

    def remove_routine_change(self, key):
        """
        remove a routine with the given name from the person's routine changes and update the routine.
        :param key: str, key to the routine change
        """

        if len(self.routine_changes) > 0: #!@
            assert key in self.routine_changes, "key " + str(key) + " not in routine changes " + str(
                self.routine_changes) + str(self)
            assert key in self.routine_change_multiplicities, "key " + str(key) + \
                                                              " not in routine change multiplicities " + str(
                self.routine_changes) + str(self)
            self.routine_change_multiplicities[key] -= 1
            if self.routine_change_multiplicities[key] == 0:
                self.routine_changes.pop(key, None)
                self.routine_change_multiplicities.pop(key, None)
                self.update_routine()

    def update_routine(self):
        """
        multiply all the routine changes, in order to get the current routine.
        """
        new_routine = {env_name: 1 for env_name in self._environments}
        for change in self.routine_changes.values():
            for env_name, val in change.items():
                assert env_name in new_routine, "environment '%s' isn't in routine %s" % (env_name, self._environments)
                new_routine[env_name] *= val
        self._current_routine = new_routine
        self._change()

    def get_city_name(self):
        """
        return person's city name
        """
        return self._environments["household"]._city.english_name.lower()

    def __repr__(self):
        return (
                'Person id: {}, age: {}, inf_prob: {}, state: {}, routine: {},' +
                'changes: {}, events: {} env: {}'
        ).format(
            self._id,
            self._age,
            self._infectiousness_prob,
            self._disease_state,
            self._current_routine,
            self.routine_changes,
            self.state_to_events,
            self._environments
        )

    def __hash__(self):
        return hash(self._id)

    @classmethod
    def person_comperator_ASCENDING(cls, a, b):
        """
        Compare persons by their age for ASCENDING sort
        """
        return a.get_age() - b.get_age()

    @classmethod
    def person_comperator_DESCENDING(cls, a, b):
        """
        Compare persons by their age for DESCENDING sort
        """
        return b.get_age() - a.get_age()
