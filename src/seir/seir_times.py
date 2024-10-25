from datetime import timedelta
from enum import Enum
import random as _random
from scipy.stats import gamma
import warnings

from src.seir.disease_state import DiseaseState
from src.simulation.params import Params
from src.util import Distribution
from src.util.Enumerations import machine_type



def daysdelta(days):
    """
    Returns a datetime.timedelta object representing 'days' full days
    :param days: The number of days to represent
    :return: datetime.timedelta(days=days)
    """
    return timedelta(days=days)


class StatisticalSeirTimesGeneration(object):
    """
    This class represents a statistical method for sampling intervals between
    disease states and probabilities of moving from each stage to the next.
    This design assumes that these intervals are independent of one another,
    and depend only on the individual.
    """
    __slots__ = ()

    def sample_latent_stage(self, person):
        """
        Returns a random latency duration and the next stage
        (could depend on the person's attributes, meaning his age)
        :param person: The Person for which we wish to sample latency duration
        :return: A pair of:
        (timedelta object which is the duration of the person's latency,
         the stage the person will go through after completing this one)
        """
        raise NotImplementedError()

    # includes asymptomatic infectious stage
    def sample_postlatent_incubation_stage(self, person):
        """
        Returns a random postlatent incubation duration and the next stage
        (could depend on the person's attributes, meaning his age)
        :param person: The Person for which we wish to sample postlatent incubation duration
        :return: A pair of:
        (timedelta object which is the duration of the person's postlatent incubation,
         the stage the person will go through after completing this one)
        """
        raise NotImplementedError()

    def sample_asymptomatic_stage(self, person):
        """
        Returns a random asymptomatic duration and the next stage
        (could depend on the person's attributes, meaning his age)
        :param person: The Person for which we wish to sample asymptomatic stage duration
        :return: A pair of:
        (timedelta object which is the duration of the person's asymptomatic stage,
         the stage the person will go through after completing this one)
        """
        raise NotImplementedError()

    def sample_symptomatic_stage(self, person):
        """
        Returns a random 'symptoms duration' and the next stage
        (could depend on the person's attributes, meaning his age)
        :param person: The Person for which we wish to sample symptoms duration
        :return: A pair of:
        (timedelta object which is the duration of the person's symptoms,
         the stage the person will go through after completing this one)
        """
        raise NotImplementedError()

    def sample_critical_stage(self, person):
        """
        Returns a random critical stage duration and the next stage
        (could depend on the person's attributes, meaning his age)
        :param person: The Person for which we wish to sample critical stage duration
        :return: A pair of:
        (timedelta object which is the duration of the person's critical stage,
         the stage the person will go through after completing this one)
        """
        raise NotImplementedError()

    def sample_stage(self, person, stage):
        """
        Gets a person and a stage (disease state) he is in,
        returns the next stage and the duration of the current stage
        (could depend on the person's attributes, meaning his age)
        :param person: The Person for which we wish to sample the stage
        :return: A pair of:
        (timedelta object which is the duration of the person's critical stage,
         the stage the person will go through after completing this one)
        """
        if stage == DiseaseState.LATENT:
            return self.sample_latent_stage(person)
        if stage == DiseaseState.INCUBATINGPOSTLATENT:
            return self.sample_postlatent_incubation_stage(person)
        if stage == DiseaseState.ASYMPTOMATICINFECTIOUS:
            return self.sample_asymptomatic_stage(person)
        if stage == DiseaseState.SYMPTOMATICINFECTIOUS:
            return self.sample_symptomatic_stage(person)
        if stage == DiseaseState.CRITICAL:
            return self.sample_critical_stage(person)

    def sample_seir_times(self, person):
        """
        Samples all stages and their durations for a given Person
        :param person: A Person for which we wish to sample SEIR stages and times
        :return: A list of (stage, duration) for its disease progression,
        where the first stage is LATENT and the final duration is None
        """
        rv = []
        curr_stage = DiseaseState.LATENT
        while curr_stage not in (DiseaseState.DECEASED, DiseaseState.IMMUNE):
            interval, next_stage = self.sample_stage(person, curr_stage)
            rv.append((curr_stage, interval))
            curr_stage = next_stage
        rv.append((curr_stage, None))
        return rv


class ConstantSeirTimesGeneration(StatisticalSeirTimesGeneration):
    """
    Naive implementation of StatisticalSeirTimesGeneration, where each stage
    lasts a constant time and the probability of passage between different
    stages does not depend on the person's age.
    This should not be used! It's here in order to provide a simple example
    of StatisticalSeirTimesGeneration and for easier debugging.
    """
    __slots__ = ()
    PROB_ASYMPTOMATIC = 0.5
    PROB_SYMPTOMATIC_TO_CRITICAL = 0.1
    PROB_CRITICAL_TO_DECEASED = 0.4  # arbitrary

    MEAN_LATENT_TIME = daysdelta(3)
    MEAN_POSTLATENT_INCUBATION_BEFORE_SYMPTOMATIC_TIME = daysdelta(1)
    MEAN_ASYMPTOMATIC_BEFORE_IMMUNE_TIME = daysdelta(7)  # asymptomatic case
    MEAN_SYMPTOMATIC_BEFORE_IMMUNE_TIME = daysdelta(7)
    MEAN_SYMPTOMATIC_BEFORE_CRITICAL_TIME = daysdelta(5)  # arbitrary
    MEAN_CRITICAL_BEFORE_DEAD_TIME = daysdelta(5)  # arbitrary
    MEAN_CRITICAL_BEFORE_IMMUNE_TIME = daysdelta(10)  # arbitrary

    def __init__(self):
        warnings.warn("DEPRECATED CODE! Using constant stage times!", DeprecationWarning)

    def sample_latent_stage(self, person):
        if _random.random() < self.PROB_ASYMPTOMATIC:
            return self.MEAN_LATENT_TIME, DiseaseState.ASYMPTOMATICINFECTIOUS
        else:
            return self.MEAN_LATENT_TIME, DiseaseState.INCUBATINGPOSTLATENT

    def sample_asymptomatic_stage(self, person):
        return self.MEAN_ASYMPTOMATIC_BEFORE_IMMUNE_TIME, DiseaseState.IMMUNE

    def sample_postlatent_incubation_stage(self, person):
        return self.MEAN_POSTLATENT_INCUBATION_BEFORE_SYMPTOMATIC_TIME, DiseaseState.SYMPTOMATICINFECTIOUS

    def sample_symptomatic_stage(self, person):
        if _random.random() < self.PROB_SYMPTOMATIC_TO_CRITICAL:
            return self.MEAN_SYMPTOMATIC_BEFORE_CRITICAL_TIME, DiseaseState.CRITICAL
        else:
            return self.MEAN_SYMPTOMATIC_BEFORE_IMMUNE_TIME, DiseaseState.IMMUNE

    def sample_critical_stage(self, person):
        if _random.random() < self.PROB_CRITICAL_TO_DECEASED:
            return self.MEAN_SYMPTOMATIC_BEFORE_CRITICAL_TIME, DiseaseState.DECEASED
        else:
            return self.MEAN_SYMPTOMATIC_BEFORE_IMMUNE_TIME, DiseaseState.IMMUNE


class RealDataSeirTimesGeneration(StatisticalSeirTimesGeneration):
    """
    Smarter implementation of StatisticalSeirTimesGeneration,
    where the duration of each stage distributes like a gamma distribution
    with some given parameters (given in params.json),
    and the probability of passage between different stages depends on the
    person's age and is also given in params.json
    """
    __slots__ = (
        'symptomatic_given_infected_per_age',
        'hospitalization_given_symptomatic_per_age',
        'critical_given_hospitalized_per_age',
        'deceased_given_critical_per_age',
        '_latent_period_distribution',
        '_infectious_before_symptomatic_distribution',
        '_infectious_before_immune_distribution',
        '_symptomatic_before_critical_distribution',
        '_symptomatic_before_immune_distribution',
        '_critical_before_deceased_distribution',
        '_critical_before_immune_distribution'
    )

    def generate_gamma_distribution(self, gamma_params):
        g = gamma(gamma_params["a"], scale=gamma_params["scale"])
        intervals = [(i, i) for i in range(1, gamma_params["max_val"] + 1)]
        probs = [g.cdf(i) - g.cdf(i - 1) for i in range(1, gamma_params["max_val"] + 1)]
        sum_probs = sum(probs)
        probs = [c / sum_probs for c in probs]
        return Distribution(intervals, probs)

    def access_table_per_age(self, table, person):
        return table[min(person.get_age() // 10, 8)]

    def prob_symptomatic(self, person):
        return self.access_table_per_age(self.symptomatic_given_infected_per_age, person)

    def prob_hospitalized_given_symptomatic(self, person):
        return self.access_table_per_age(self.hospitalization_given_symptomatic_per_age, person)

    def prob_critical_given_hospitalized(self, person):
        return self.access_table_per_age(self.critical_given_hospitalized_per_age, person)

    def prob_deceased_given_critical(self, person):
        return self.access_table_per_age(self.deceased_given_critical_per_age, person)

    def __init__(self):
        params = Params.loader()['disease_parameters']
        self.symptomatic_given_infected_per_age = params["symptomatic_given_infected_per_age"]
        self.hospitalization_given_symptomatic_per_age = params["hospitalization_given_symptomatic_per_age"]
        self.critical_given_hospitalized_per_age = params["critical_given_hospitalized_per_age"]
        self.deceased_given_critical_per_age = params["deceased_given_critical_per_age"]

        self._latent_period_distribution = self.generate_gamma_distribution(
            params["latent_period_gamma_params"]
        )
        self._infectious_before_symptomatic_distribution = self.generate_gamma_distribution(
            params["infectious_before_symptomatic_gamma_params"]
        )
        self._infectious_before_immune_distribution = self.generate_gamma_distribution(
            params["infectious_before_immune_gamma_params"]
        )
        self._symptomatic_before_critical_distribution = self.generate_gamma_distribution(
            params["symptomatic_before_critical_gamma_params"]
        )
        self._symptomatic_before_immune_distribution = self.generate_gamma_distribution(
            params["symptomatic_before_immune_gamma_params"]
        )
        self._critical_before_deceased_distribution = self.generate_gamma_distribution(
            params["critical_before_deceased_gamma_params"]
        )
        self._critical_before_immune_distribution = self.generate_gamma_distribution(
            params["critical_before_immune_gamma_params"]
        )

    def sample_bool(self, prob):
        """
        returns True in probability 'prob'
        """
        return _random.random() < prob

    def sample_latent_stage(self, person):
        tmpprob = self.prob_symptomatic(person)
        if self.sample_bool(tmpprob):
            return daysdelta(self._latent_period_distribution.sample()), DiseaseState.INCUBATINGPOSTLATENT
        else:
            return daysdelta(self._latent_period_distribution.sample()), DiseaseState.ASYMPTOMATICINFECTIOUS

    # includes asymptomatic infectious stage
    def sample_postlatent_incubation_stage(self, person):
        return daysdelta(self._infectious_before_symptomatic_distribution.sample()), DiseaseState.SYMPTOMATICINFECTIOUS

    def sample_asymptomatic_stage(self, person):
        return daysdelta(self._infectious_before_immune_distribution.sample()), DiseaseState.IMMUNE

    def sample_symptomatic_stage(self, person):
        critical_given_symptomatic = \
            self.prob_critical_given_hospitalized(person) * \
            self.prob_hospitalized_given_symptomatic(person)
        if self.sample_bool(critical_given_symptomatic):
            return daysdelta(self._symptomatic_before_critical_distribution.sample()), DiseaseState.CRITICAL
        else:
            return daysdelta(self._symptomatic_before_immune_distribution.sample()), DiseaseState.IMMUNE

    def sample_critical_stage(self, person):
        if self.sample_bool(self.prob_deceased_given_critical(person)):
            return daysdelta(self._critical_before_deceased_distribution.sample()), DiseaseState.DECEASED
        else:
            return daysdelta(self._critical_before_immune_distribution.sample()), DiseaseState.IMMUNE

    # For singleton use:
    singleton = None
    @classmethod
    def make(cls):
        """
        Use this instead of the constructor of the function in order to only
        construct this object once (otherwise this would be wasteful in
        running time, no reason not to do it)
        :return: A RealDataSeirTimesGeneration object
        """
        if cls.singleton is None:
            cls.singleton = cls()
        return cls.singleton
    
    @classmethod
    def clean(cls):
        """
        Clean the value of the singleton in case we want to update it's values
        """
        cls.singleton = None

class SIRS(RealDataSeirTimesGeneration):
    """
    Smarter implementation of RealDataSeirTimesGeneration,
    where instead of SIR model, we will implement SIRS model where 
    a person from "Removed" contpartment will go again to "Susceptible"
    contpatment
    """
    __slots__ =('susceptible_given_immuned_per_age',
                '_immuned_before_susceptible_distribution')
    
    def __init__(self):
        super().__init__()
        params = Params.loader()['disease_parameters']
        self._immuned_before_susceptible_distribution = self.generate_gamma_distribution(
            params["Immuned_before_susceptible_gamma_params"]
        )
    
    def prob_susptible_given_Immuned(self,person):
        return self.access_table_per_age(self.susceptible_given_immuned_per_age,person)

    def sample_immune_stage(self, person):
        #on 25/8/21 Ori asked that on SIRS that Immuned people will became susptible after some time
        return daysdelta(self._immuned_before_susceptible_distribution.sample()), DiseaseState.SUSCEPTIBLE
    def sample_stage(self, person, stage):
        """
        Gets a person and a stage (disease state) he is in,
        returns the next stage and the duration of the current stage
        (could depend on the person's attributes, meaning his age)
        :param person: The Person for which we wish to sample the stage
        :return: A pair of:
        (timedelta object which is the duration of the person's critical stage,
         the stage the person will go through after completing this one)
        """
        if stage == DiseaseState.LATENT:
            return self.sample_latent_stage(person)
        if stage == DiseaseState.INCUBATINGPOSTLATENT:
            return self.sample_postlatent_incubation_stage(person)
        if stage == DiseaseState.ASYMPTOMATICINFECTIOUS:
            return self.sample_asymptomatic_stage(person)
        if stage == DiseaseState.SYMPTOMATICINFECTIOUS:
            return self.sample_symptomatic_stage(person)
        if stage == DiseaseState.CRITICAL:
            return self.sample_critical_stage(person)
        if stage == DiseaseState.IMMUNE:
            return self.sample_immune_stage(person)

    def sample_seir_times(self, person):
        """
        Samples all stages and their durations for a given Person
        :param person: A Person for which we wish to sample SEIRS stages and times
        :return: A list of (stage, duration) for its disease progression,
        where the first stage is LATENT and the final duration is None
        """
        rv = []
        curr_stage = DiseaseState.LATENT
        while curr_stage not in (DiseaseState.DECEASED, DiseaseState.SUSCEPTIBLE):
            interval, next_stage = self.sample_stage(person, curr_stage)
            rv.append((curr_stage, interval))
            curr_stage = next_stage
        rv.append((curr_stage, None))
        return rv

    
    @classmethod
    def make(cls):
        """
        Use this instead of the constructor of the function in order to only
        construct this object once (otherwise this would be wasteful in
        running time, no reason not to do it)
        :return: A SIRS object
        """
        if cls.singleton is None:
            cls.singleton = cls()
        return cls.singleton

def sample_seir_times(sir_type:machine_type ,person):
    """
    Samples and returns the SEIR stages and durations for a given Person
    (depends on his age)
    :param person: A Person for which we wish to sample SEIR stages and times
    :return: A list of (stage, duration) for its disease progression,
    where the first stage is LATENT and the final duration is None
    """
       
    if sir_type == machine_type.SIR:
        return RealDataSeirTimesGeneration.make().sample_seir_times(person)
    elif sir_type == machine_type.SIRS:
        return SIRS.make().sample_seir_times(person)
