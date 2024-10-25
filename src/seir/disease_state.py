from enum import Enum
from src.simulation.params import Params
from datetime import date


class InfectiousnessFactors(object):
    """
    A singleton class which loads the infectiousness of each DiseaseState
    """
    __slots__ = ()
    infectiousness_factors = None

    @classmethod
    def get_infectiousness_map(cls):
        if cls.infectiousness_factors is None:
            factors = Params.loader()["disease_parameters"]["infectiousness_per_stage"]
            cls.infectiousness_factors = {
                DiseaseState.INCUBATINGPOSTLATENT: factors["incubating_post_latent"],
                DiseaseState.ASYMPTOMATICINFECTIOUS: factors["asymptomatic"],
                DiseaseState.SYMPTOMATICINFECTIOUS: factors["symptomatic"],
                DiseaseState.CRITICAL: factors["critical"]
            }
        return cls.infectiousness_factors


class DiseaseState(Enum):
    """
    An enum class
    (meaning its objects have some predetermined list of possible values),
    describing the different possible disease states people could have.
    """
    __slots__ = ()
    # The order matters for efficiency (see the implementation of is_infected)
    INCUBATINGPOSTLATENT = 1    # Infectious pre-symptomatic
    ASYMPTOMATICINFECTIOUS = 2  # Infectious asymptomatic
    SYMPTOMATICINFECTIOUS = 3   # Infectious and symptomatic
    CRITICAL = 4                # Infectious and requires critical care
    LATENT = 5                  # Latent (not infectious yet, but about to be)
    SUSCEPTIBLE = 6             # Not infected but prone to infection
    DECEASED = 7                # No longer among the living
    IMMUNE = 8                  # Has been infected before, recovered, no longer prone to infection
    #change_date = None   # !+
    VACCINATED = 9  #NEW

    @classmethod
    def init_infectiousness_list(cls):
        """
        Static function which loads the infectiousness factors
        so we may access it from the enum value.
        """
        cls.infectiousness_list = [0.]
        for ind in range(1, cls.IMMUNE._value_ + 1):
            if ind <= cls.CRITICAL._value_:
                cls.infectiousness_list.append(
                    InfectiousnessFactors.get_infectiousness_map()[DiseaseState(ind)]
                )
            else:
                cls.infectiousness_list.append(0)

    def get_infectiousness_factor(self):
        """
        :return: The infectiousness factor of self (some disease state)
        """
        return DiseaseState.infectiousness_list[self._value_]

    def is_infected(self):
        """
        :return: Is this disease state corresponding to a currently infected
        individual
        (ASYMPTOMATICINFECTIOUS, INCUBATINGPOSTLATENT, SYMPTOMATICINFECTIOUS,
        CRITICAL or LATENT)
        """
        return self._value_ <= 5

    def is_infectious(self):
        """
        :return: Is this disease state corresponding to a currently infectious
        individual
        (ASYMPTOMATICINFECTIOUS, INCUBATINGPOSTLATENT, SYMPTOMATICINFECTIOUS or
        CRITICAL)
        """
        # ASYMPTOMATICINFECTIOUS, INCUBATINGPOSTLATENT, SYMPTOMATICINFECTIOUS or CRITICAL
        return self._value_ <= 4

    def is_exposed(self):
        """
        Returns whether this disease state may no longer be advanced
        :return: self in (DECEASED, IMMUNE)
        """
        #INCUBATINGPOSTLATENT, ASYMPTOMATICINFECTIOUS, SYMPTOMATICINFECTIOUS, and LATENT
        return self in (DiseaseState.INCUBATINGPOSTLATENT,
                        DiseaseState.ASYMPTOMATICINFECTIOUS,
                        DiseaseState.SYMPTOMATICINFECTIOUS,
                        DiseaseState.LATENT)

    def is_critical(self):
        """
        :return: self == CRITICAL
        """
        return self == DiseaseState.CRITICAL

    def is_dead(self):
        """
        :return: self == DECEASED
        """
        return self == DiseaseState.DECEASED

    def is_static(self):
        """
        :return: Is this disease state corresponding to not infected individuals
        (SUSCEPTIBLE, IMMUNE or DECEASED)
        """
        return not self.is_infected()

    def is_symptomatic(self):
        """
        :return: self in (SYMPTOMATICINFECTIOUS, CRITICAL)
        """
        return self in (
            DiseaseState.SYMPTOMATICINFECTIOUS,
            DiseaseState.CRITICAL
        )

    def is_susceptible(self):
        """
        :return: self == SUSCEPTIBLE
        """
        return self == DiseaseState.SUSCEPTIBLE

    def is_terminal(self):
        """
        Returns whether this disease state may no longer be advanced
        :return: self in (DECEASED, IMMUNE)
        """
        return self in (DiseaseState.DECEASED, DiseaseState.IMMUNE)

    def is_vaccinated(self):
        return self == DiseaseState.VACCINATED