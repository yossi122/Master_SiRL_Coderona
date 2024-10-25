import datetime

class Trigger:
    """
    Represents a condition that should be checked before applying an event.
    It accepts a simulation and returns either True or False.
    Each Trigger subclass implements the trigger() func
    """
    __slots__ = ('_is_triggered',)

    def __init__(self):
        self._is_triggered = False

    def trigger(self, simulation):
        """
        returns whether the trigger's condition holds
        :param simulation: Simulation
        :return: bool
        """
        raise NotImplementedError()

    def __and__(self, other):
        return AndTrigger(self, other)

    def __or__(self, other):
        return OrTrigger(self, other)


class EmptyTrigger(Trigger):
    def __init__(self):
        super().__init__()

    def trigger(self, simulation):
        """
        returns True always
        :param simulation: Simulation object
        :return: True
        """
        res = True
        self._is_triggered |= res
        return res


class DayTrigger(Trigger):
    """
    This trigger is True if the simulation is on the same date as the date saved on this object.
    """
    __slots__ = ('date',)

    def __init__(self, date):
        """
        Init a trigger that is dependent on the given date
        :param date: datetime.date date object
        """
        super().__init__()
        assert isinstance(date, datetime.date)
        self.date = date

    def trigger(self, simulation):
        """
        returns whether the trigger's condition holds
        :param simulation: Simulation object
        :return: True if the simulation is on the same date as the date saved on the trigger
        """
        res = simulation._date == self.date
        self._is_triggered |= res
        return res


class TimeRangeTrigger(Trigger):
    """
    This trigger is True if the simulation's date is between the dates saved on the trigger,
    meaning in the time range (start_date, end_date).
    """
    __slots__ = ('start_date', 'end_date')

    def __init__(self, start_date, end_date):
        """
        Init the trigger with time range dates, so the time range is (start_date, end_date)
        :param start_date: datetime.date
        :param end_date: datetime.date
        """
        super().__init__()
        assert isinstance(start_date, datetime.date)
        assert isinstance(end_date, datetime.date)
        self.start_date = start_date
        self.end_date = end_date

    def trigger(self, simulation):
        """
        returns whether the trigger's condition holds
        :param simulation: Simulation object
        :return: True if the date on the simulation is in the range (start_date, end_date)
        """
        res = self.start_date <= simulation._date < self.end_date
        self._is_triggered |= res
        return res


class AfterTrigger(Trigger):
    """
    This trigger checks whether some given event has been applied.
    That way, an event that has this trigger cannot be applied before the trigger's self.event
    """
    __slots__ = ('_event',)

    def __init__(self, event):
        super().__init__()
        self._event = event

    def trigger(self, simulation):
        """
        returns whether the trigger's condition holds
        :param simulation: Simulation object
        :return: True if self.event is applied
        """
        res = self._event.is_applied
        self._is_triggered |= res
        return res


class MultiTrigger(Trigger):
    """
    An abstract subclass, that represents a trigger that has many sub triggers to check.
    """
    __slots__ = ('triggers',)

    def __init__(self, triggers, *args):
        super().__init__()
        if not isinstance(triggers, list):
            triggers = [triggers]
        self.triggers = triggers + list(args)


class OrTrigger(MultiTrigger):
    """
    This multi trigger is true when at least one of its sub triggers is true
    """
    def __init__(self, triggers, *args):
        super().__init__(triggers, *args)

    def trigger(self, simulation):
        """
        returns whether the trigger's condition holds
        :param simulation: Simulation object
        :return: True if any of the sub triggers are true
        """
        res = any(trig.trigger(simulation) for trig in self.triggers)
        self._is_triggered |= res
        return res


class AndTrigger(MultiTrigger):
    """
    This multi trigger is true is all of its sub triggers are true
    """
    def __init__(self, triggers, *args):
        super().__init__(triggers, *args)

    def trigger(self, simulation):
        """
        returns whether the trigger's condition holds
        :param simulation: Simulation object
        :return: True if all of the sub triggers are true
        """
        res = all(trig.trigger(simulation) for trig in self.triggers)
        self._is_triggered |= res
        return res


class EmptyEffect:
    """
    Does nothing.
    Useful when one wishes to create an event to hook other events to (without that event doing anything itself).
    """
    def __init__(self):
        pass

    def apply(self, simulation):
        pass


class DiseaseStateChangeEffect:
    """
    Effect that changes a Person's disease state
    """
    __slots__ = ('_person', '_old_state', '_new_state')

    def __init__(self, person, old_state, new_state):
        """
        :param person: Person
        :param old_state: DiseaseState
        :param new_state: DiseaseState
        """
        self._person = person
        self._old_state = old_state
        self._new_state = new_state
        # print("Creating DiseaseStateChangeEffect id:{} old:{} new:{}".format(person.get_id(),old_state,new_state))

    def apply(self, simulation):
        """
        Change the disease state from old_state to new_state
        :param simulation: Simulation object
        """
        # print("change person:{} from:{} to:{}".format(self._person.get_id(),self._old_state,self._new_state))
        #!@
        if self._person.get_disease_state() == self._old_state:
            assert self._person.get_disease_state() == self._old_state, (
                str(self._person.get_disease_state()) +
                " - " + str(self._old_state) +
                " of id " + str(self._person._id)
            )
            self._person.set_disease_state(self._new_state)
        # else:
        #     print(f"event.DiseaseStateChangeEffect.apply() ignoring change to {str(self._new_state)} as person is now in {str(self._person.get_disease_state())}")

    def get_person(self):
        return self._person

    def get_states(self):
        return self._old_state, self._new_state


class DelayedEffect:
    """
    This effect is used to apply some event delay_time time after the moment it is applied.
    It means that instead of being applied when the event's trigger is True, the given event will apply after a delay
    """
    __slots__ = ('_event', '_delay_time')

    def __init__(self, event, delay_time):
        """
        :param event: Event object to be delayed
        :param delay_time: timedelta to represent days
        """
        assert isinstance(delay_time, datetime.timedelta)
        self._event = event
        self._delay_time = delay_time

    def apply(self, simulation):
        """
        register self.event to happen in delay_time days
        :param simulation: Simulation object
        """
        target_day = simulation._date + self._delay_time
        simulation.register_event_on_day(self._event, target_day)


class AddRoutineChangeEffect:
    """
    An effect that adds some routine change to some Person
    """
    __slots__ = ('_person', 'routine_change_key', 'routine_change_val')

    def __init__(self, person, routine_change_key, routine_change_val):
        """
        :param person: Person
        :param routine_change_key: str The name of the routine change
        :param routine_change_val: the new routine change
        """
        self._person = person
        self.routine_change_key = routine_change_key
        self.routine_change_val = routine_change_val

    def apply(self, simulation):
       # print('AddRoutineChangeEffect', self.routine_change_key, self.routine_change_val)
        # !@ For some reasons occuring gaps - no change key etc - # TODO: its probably fixes crush
        if self.routine_change_key and self.routine_change_val:
            self._person.add_routine_change(self.routine_change_key, self.routine_change_val)


class AddRoutineChangeEnvironmentEffect:
    """
    An effect which adds some routine change to all people in some environment
    """
    __slots__ = ('_environment', 'routine_change_key', 'routine_change_generator')

    def __init__(self, environment, routine_change_key, routine_change_generator):
        self._environment = environment
        self.routine_change_key = routine_change_key
        self.routine_change_generator = routine_change_generator

    def apply(self, simulation):
        for member in self._environment.get_people():
           # print('AddRoutineChangeEnvironmentEffect', self.routine_change_key, self.routine_change_generator)
            member.add_routine_change(self.routine_change_key, self.routine_change_generator(member))

class ChangeEnvInterventionStateEffect:
    """
    An effect which changes the intervention state of some environment
    """
    __slots__ = ('_env', '_old_state', '_new_state')

    def __init__(self, env, old_state, new_state):
        self._env = env
        self._old_state = old_state
        self._new_state = new_state

    def apply(self, simulation):
       # print(self._old_state, self._new_state)
        self._env.set_intervention_state(self._old_state, self._new_state)

class RemoveRoutineChangeEnvironmentEffect:
    """
    An effect which removes some routine change from all people in some environment
    """
    __slots__ = ('_environment', 'routine_change_key')

    def __init__(self, environment, routine_change_key):
        self._environment = environment
        self.routine_change_key = routine_change_key

    def apply(self, simulation):
        for member in self._environment.get_people():
            member.remove_routine_change(self.routine_change_key)


class RemoveRoutineChangeEffect:
    """
    An effect which removes a routine change from a person
    """
    __slots__ = ('_person', 'routine_change_key',)

    def __init__(self, person, routine_change_key):
        self._person = person
        self.routine_change_key = routine_change_key

    def apply(self, simulation):
        if self.routine_change_key:
            #print('RemoveRoutineChangeEffect', self.routine_change_key)
            self._person.remove_routine_change(self.routine_change_key)


class _Hookable():
    """
    The interface of hooks, which are lists of other events that should be applied after a successful application of some trigger.
    """
    __slots__ = ('hooks',)

    def __init__(self):
        self.hooks = []

    def _hook_one(self, sub):
        """
        Hooks some event onto self
        :param sub: Event
        """
        if isinstance(sub, DayEvent) and isinstance(sub.effect, EmptyEffect):
            assert isinstance(self, DayEvent) and self._date == sub._date, "failed to flatten an empty DayEvent"
            self.hook(sub.hooks)
        else:
            self.hooks.append(sub)

    def hook(self, sub_events):
        """
        hook the given events on this event
        :param sub_events: one or more events
        """
        if isinstance(sub_events, _Hookable):
            self._hook_one(sub_events)
            return
        for e in sub_events:
            self._hook_one(e)

    def hooks_apply(self, simulation):
        """
        Applies all the events that are hooked on this event
        :param simulation: Simulation object to pass to all hooked events
        """
        for h in self.hooks:
            h.apply(simulation)

    def get_tree(self):
        """
        recursively gets a tree of the events hooked onto this events, their hooked events and so on.
        :return: List of all hooks
        """
        tree = [self]
        for h in self.hooks:
            tree += h.get_tree()
        return tree


class Event(_Hookable):
    """
    Event is an object that can be applied, only when some conditions are true.
    This class implement the general logic of "if X, the do Y" in the simulation.
    Each event object has its trigger and effect. The trigger, hold the "if X" logic, and the effect if the "do Y".
    Also, each Event can have hooks, which are other events that need to be applied afterwards.
    """
    __slots__ = ('trigger', 'effect', 'is_applied')

    def __init__(self, trigger=None, effect=None):
        """
        initialize the event
        :param trigger: Trigger object, if None - EmptyTrigger will be used
        :param effect: Effect object, if None - EmptyEffect
        """
        super().__init__()
        # assert trigger or effect, 'Why do you need meaningless event'
        if trigger is None:
            trigger = EmptyTrigger()
        if effect is None:
            effect = EmptyEffect()
        self.trigger = trigger
        self.effect = effect
        self.is_applied = False

    def apply(self, simulation):
        """
        Activate the event if the trigger is True (then applies all of its hooks).
        Calls the apply func of the event's effect.
        This can happen only once!
        :param simulation: Simulation
        """
        if not self.trigger.trigger(simulation):
            return
        # assert not self.is_applied
        if not self.is_applied:
            self.is_applied = True
            self.effect.apply(simulation)
            self.hooks_apply(simulation)


class DayEvent(Event):
    """
    A special subclass of Event that is triggered by a DayTrigger.
    """
    __slots__ = ('_date',)

    def __init__(self, date, effect=None):
        if effect is None:
            effect = EmptyEffect()
        super().__init__(DayTrigger(date), effect)
        self._date = date
