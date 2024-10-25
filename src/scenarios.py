from datetime import timedelta

from src.run_utils import INITIAL_DATE
from src.seir import daysdelta
from src.seir.disease_state import DiseaseState
from src.simulation.interventions import *
from src.simulation.interventions.intervention import LockdownIntervention
from src.world.environments.environment import EnvironmentalAttribute


# first scenarios
def scenario_1_interventions(compliance, ci_delay, hi_delay):
    ci_intervention = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        delay=ci_delay
    )
    sd_intervention = SocialDistancingIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        age_range=(0, 99)
    )
    school_closer_intervention = SchoolClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(60),
        compliance=1.0,
        proportion_of_envs=1.0,
        city_name='all',
        age_segment=(3, 22)
    )
    workplace_closure_intervention = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(60),
        compliance=0.85
    )
    elderly_intervention = ElderlyQuarantineIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        min_age=70
    )
    household_intervention = HouseholdIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        delay_on_enter=hi_delay
    )
    interventions = [ci_intervention,
                     sd_intervention,
                     workplace_closure_intervention,
                     elderly_intervention,
                     household_intervention,
                     school_closer_intervention
                     ]
    return interventions


def noam_lockdown_scenario(compliance=1, ci_delay=None, hi_delay=None):
    lockdown_intervention = LockdownIntervention(
        start_date=INITIAL_DATE + timedelta(0.0),
        duration=daysdelta(4 * 7),
        compliance=0.80,
        city_name='all'
    )
    interventions = [
        lockdown_intervention
    ]
    return interventions


def vaccinations_scenario_general(compliance=1, ci_delay=None, hi_delay=None):
    vaccinations = ImmuneGeneralPopulationIntervention(
        compliance=compliance,
        start_date=INITIAL_DATE,
        duration=daysdelta(16 * 7),
        people_per_day=100,
        min_age=18)
    interventions = [
        vaccinations
    ]
    return interventions


def vaccinations_scenario_households(compliance=1, ci_delay=None, hi_delay=None):
    vaccinations = ImmuneByHouseholdIntervention(
        compliance=compliance,
        start_date=INITIAL_DATE,
        duration=daysdelta(16 * 7),
        houses_per_day=20,
        min_age=18)
    interventions = [
        vaccinations
    ]
    return interventions


def Empty_scenario(compliance=1, ci_delay=None, hi_delay=None):
    return []


def scenario_21_interventions(compliance, ci_delay, hi_delay):
    ci_intervention = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        delay=ci_delay
    )
    sd_intervention = SocialDistancingIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        age_range=(0, 99)
    )
    school_closer_intervention = SchoolClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(60),
        compliance=1.0,
        proportion_of_envs=0.98,
        city_name='all',
        age_segment=(3, 22)
    )
    workplace_closure_intervention = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(60),
        compliance=0.7
    )
    elderly_intervention = ElderlyQuarantineIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        min_age=65
    )
    household_intervention = HouseholdIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        delay_on_enter=hi_delay
    )
    interventions = [ci_intervention,
                     sd_intervention,
                     workplace_closure_intervention,
                     elderly_intervention,
                     household_intervention,
                     school_closer_intervention
                     ]
    return interventions


def scenario_22_interventions(compliance, ci_delay, hi_delay):
    ci_intervention = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        delay=ci_delay
    )
    sd_intervention = SocialDistancingIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        age_range=(0, 99)
    )
    school_closer_intervention = SchoolClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(60),
        compliance=1.0,
        proportion_of_envs=0.98,
        city_name='all',
        age_segment=(3, 17)
    )
    school_closer_intervention_yeshiva = SchoolClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(60),
        compliance=1.0,
        proportion_of_envs=0.5,
        city_name='all',
        age_segment=(18, 22)
    )
    school_isolation_intervention_yeshiva = SchoolIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(60),
        compliance=1.0,
        proportion_of_envs=0.5,
        city_name='all',
        age_segment=(18, 22)
    )
    workplace_closure_intervention = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(60),
        compliance=0.70
    )
    elderly_intervention = ElderlyQuarantineIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        min_age=65
    )
    household_intervention = HouseholdIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        delay_on_enter=hi_delay
    )
    interventions = [ci_intervention,
                     sd_intervention,
                     workplace_closure_intervention,
                     elderly_intervention,
                     household_intervention,
                     school_closer_intervention,
                     school_isolation_intervention_yeshiva,
                     school_closer_intervention_yeshiva
                     ]
    return interventions


def scenario_23_interventions(compliance, ci_delay, hi_delay):
    ci_intervention = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        delay=ci_delay
    )
    sd_intervention = SocialDistancingIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        age_range=(0, 99)
    )
    school_closer_intervention = SchoolClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(60),
        compliance=1.0,
        proportion_of_envs=0.98,
        city_name='all',
        age_segment=(10, 22)
    )
    workplace_closure_intervention = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(60),
        compliance=0.70
    )
    elderly_intervention = ElderlyQuarantineIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        min_age=65
    )
    household_intervention = HouseholdIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(360),
        compliance=compliance,
        delay_on_enter=hi_delay
    )
    interventions = [ci_intervention,
                     sd_intervention,
                     workplace_closure_intervention,
                     elderly_intervention,
                     household_intervention,
                     school_closer_intervention
                     ]
    return interventions


def scenario_24_interventions(compliance, ci_delay, hi_delay):
    ci_intervention = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        delay=ci_delay
    )
    sd_intervention = SocialDistancingIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        age_range=(0, 99)
    )
    school_closer_intervention = SchoolClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(60),
        compliance=1.0,
        proportion_of_envs=0.98,
        city_name='all',
        age_segment=(10, 17)
    )
    school_closer_intervention_yeshiva = SchoolClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(60),
        compliance=1.0,
        proportion_of_envs=0.5,
        city_name='all',
        age_segment=(18, 22)
    )
    school_isolation_intervention_yeshiva = SchoolIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(60),
        compliance=1.0,
        proportion_of_envs=0.5,
        city_name='all',
        age_segment=(18, 22)
    )
    workplace_closure_intervention = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(60),
        compliance=0.70
    )
    elderly_intervention = ElderlyQuarantineIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        min_age=65
    )
    household_intervention = HouseholdIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        delay_on_enter=hi_delay
    )
    interventions = [ci_intervention,
                     sd_intervention,
                     workplace_closure_intervention,
                     elderly_intervention,
                     household_intervention,
                     school_closer_intervention,
                     school_closer_intervention_yeshiva,
                     school_isolation_intervention_yeshiva
                     ]
    return interventions


def scenario_25_interventions(compliance, ci_delay, hi_delay):
    ci_intervention = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        delay=ci_delay
    )
    sd_intervention = SocialDistancingIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        age_range=(0, 99)
    )
    school_closer_intervention = SchoolClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(60),
        compliance=1.0,
        proportion_of_envs=0.98,
        city_name='all',
        age_segment=(6, 17)
    )
    school_closer_intervention_yeshiva = SchoolClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(60),
        compliance=1.0,
        proportion_of_envs=0.5,
        city_name='all',
        age_segment=(18, 22)
    )
    school_isolation_intervention_yeshiva = SchoolIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(60),
        compliance=1.0,
        proportion_of_envs=0.5,
        city_name='all',
        age_segment=(18, 22)
    )
    workplace_closure_intervention = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(60),
        compliance=0.70
    )
    elderly_intervention = ElderlyQuarantineIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        min_age=65
    )
    household_intervention = HouseholdIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        delay_on_enter=hi_delay
    )
    interventions = [ci_intervention,
                     sd_intervention,
                     workplace_closure_intervention,
                     elderly_intervention,
                     household_intervention,
                     school_closer_intervention,
                     school_closer_intervention_yeshiva,
                     school_isolation_intervention_yeshiva
                     ]
    return interventions


def scenario_26_interventions(compliance, ci_delay, hi_delay):
    ci_intervention = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        delay=ci_delay
    )
    sd_intervention = SocialDistancingIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        age_range=(0, 99)
    )
    periodic_school_closure_intervention = SchoolClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(60),
        compliance=1.0,
        proportion_of_envs=1.0,
        city_name='all',
        age_segment=(3, 9),
        period_data=AttributeAndPeriodData(
            EnvironmentalAttribute('household', 'last name', 2),
            timedelta(7)
        )
    )
    school_closer_intervention = SchoolClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(60),
        compliance=1.0,
        proportion_of_envs=0.98,

        city_name='all',
        age_segment=(10, 22)
    )
    workplace_closure_intervention = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(60),
        compliance=0.70
    )
    elderly_intervention = ElderlyQuarantineIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        min_age=65
    )
    household_intervention = HouseholdIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        delay_on_enter=hi_delay
    )
    interventions = [ci_intervention,
                     sd_intervention,
                     workplace_closure_intervention,
                     elderly_intervention,
                     household_intervention,
                     school_closer_intervention,
                     periodic_school_closure_intervention
                     ]
    return interventions


def householdisolation_sd_interventions(compliance, ci_delay, hi_delay):
    ci_intervention = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(360),
        compliance=1,
        delay=ci_delay
    )
    sd_intervention = SocialDistancingIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(360),
        compliance=0.8,
        age_range=(0, 99)
    )
    household_intervention = HouseholdIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(360),
        compliance=1,
        delay_on_enter=hi_delay
    )
    interventions = [sd_intervention,
                     ci_intervention,
                     household_intervention
                     ]
    return interventions


def old_specific_HH_interventions(compliance, ci_delay, hi_delay):
    ci_intervention = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(120),
        compliance=1,
        delay=ci_delay
    )
    sd_intervention = SocialDistancingIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(120),
        compliance=compliance,
        age_range=(0, 99)
    )
    household_intervention = HouseholdIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(120),
        compliance=compliance,
        delay_on_enter=hi_delay
    )
    elderly_intervention = ElderlyQuarantineIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(120),
        compliance=1,
        min_age=60
    )
    interventions = [sd_intervention,
                     ci_intervention,
                     household_intervention,
                     elderly_intervention
                     ]
    return interventions


def old_specific_noHH_interventions(compliance, ci_delay, hi_delay):
    ci_intervention = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(120),
        compliance=compliance,
        delay=ci_delay
    )
    sd_intervention = SocialDistancingIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(120),
        compliance=compliance,
        age_range=(0, 99)
    )
    household_intervention = HouseholdIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(120),
        compliance=compliance,
        delay_on_enter=hi_delay
    )
    elderly_intervention = ElderlyQuarantineIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(120),
        compliance=0.9,
        min_age=60
    )
    interventions = [sd_intervention,
                     elderly_intervention
                     ]
    return interventions


def adult_specific_HH_interventions(compliance, ci_delay, hi_delay):
    ci_intervention = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(120),
        compliance=1,
        delay=ci_delay
    )
    sd_intervention = SocialDistancingIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(120),
        compliance=compliance,
        age_range=(0, 99)
    )
    household_intervention = HouseholdIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(120),
        compliance=compliance,
        delay_on_enter=hi_delay
    )
    workplace_closure_intervention = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(120),
        compliance=0.75
    )
    interventions = [sd_intervention,
                     ci_intervention,
                     household_intervention,
                     workplace_closure_intervention
                     ]
    return interventions


def adult_specific_noHH_interventions(compliance, ci_delay, hi_delay):
    ci_intervention = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(120),
        compliance=compliance,
        delay=ci_delay
    )
    sd_intervention = SocialDistancingIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(120),
        compliance=compliance,
        age_range=(0, 99)
    )
    household_intervention = HouseholdIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(120),
        compliance=compliance,
        delay_on_enter=hi_delay
    )
    workplace_closure_intervention = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(120),
        compliance=1
    )
    interventions = [sd_intervention,
                     workplace_closure_intervention
                     ]
    return interventions


def children_school_closure_intervention(compliance, ci_delay, hi_delay):
    ci_intervention = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(120),
        compliance=1,
        delay=ci_delay
    )
    sd_intervention = SocialDistancingIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(120),
        compliance=0.8,
        age_range=(0, 99)
    )
    household_intervention = HouseholdIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(120),
        compliance=1,
        delay_on_enter=hi_delay
    )
    school_closer_intervention = SchoolClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(120),
        compliance=1.0,
        proportion_of_envs=1,
        city_name='all',
        age_segment=(6, 18)
    )
    interventions = [sd_intervention,
                     ci_intervention,
                     household_intervention,
                     school_closer_intervention
                     ]
    return interventions


def children_asymptomatic_detection_intervention(compliance, ci_delay, hi_delay):
    ci_intervention = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(360),
        compliance=1,
        delay=ci_delay
    )
    # to simulate detection of asymptomatic children at school
    asi_intervention = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(360),
        compliance=compliance,
        delay=0,
        min_age=6,
        max_age=18,
        entry_states=(
            DiseaseState.LATENT,
            DiseaseState.ASYMPTOMATICINFECTIOUS
        )
    )
    household_intervention = HouseholdIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(360),
        compliance=1,
        delay_on_enter=hi_delay
    )
    household_intervention_of_asymptomatic = HouseholdIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(360),
        compliance=compliance,
        delay_on_enter=hi_delay,
        min_age=6,
        max_age=18,
        entry_states=(
            DiseaseState.LATENT,
            DiseaseState.ASYMPTOMATICINFECTIOUS
        )
    )
    sd_intervention = SocialDistancingIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(360),
        compliance=0.8,
        age_range=(0, 99)
    )
    interventions = [
        sd_intervention,
        ci_intervention,
        household_intervention,
        #household_intervention_of_asymptomatic,
        asi_intervention
    ]
    return interventions


def only_children_asymptomatic_detection(compliance, ci_delay, hi_delay):
    # to simulate detection of asymptomatic children at school
    asi_intervention = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(120),
        compliance=0.7,
        delay=0,
        min_age=6,
        max_age=18,
        entry_states=(
            DiseaseState.LATENT,
            DiseaseState.ASYMPTOMATICINFECTIOUS
        )
    )

    interventions = [asi_intervention]
    return interventions


def children_specific_noHH_interventions(compliance, ci_delay, hi_delay):
    ci_intervention = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(120),
        compliance=compliance,
        delay=ci_delay
    )
    sd_intervention = SocialDistancingIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(120),
        compliance=compliance,
        age_range=(0, 99)
    )
    household_intervention = HouseholdIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(120),
        compliance=compliance,
        delay_on_enter=hi_delay
    )
    school_closer_intervention = SchoolClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(120),
        compliance=1.0,
        proportion_of_envs=1,
        city_name='all',
        age_segment=(6, 18)
    )
    interventions = [sd_intervention,
                     school_closer_intervention
                     ]
    return interventions


def scenario_232_interventions(compliance, ci_delay, hi_delay):
    ci_intervention = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        delay=ci_delay
    )
    sd_intervention = SocialDistancingIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        age_range=(0, 99)
    )
    school_closer_intervention = SchoolClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(60),
        compliance=1.0,
        proportion_of_envs=0.98,
        city_name='all',
        age_segment=(10, 22)
    )
    workplace_closure_intervention = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(60),
        compliance=0.70
    )
    elderly_intervention = ElderlyQuarantineIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        min_age=65
    )
    household_intervention = HouseholdIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        delay_on_enter=hi_delay
    )
    interventions = [ci_intervention,
                     sd_intervention,
                     workplace_closure_intervention,
                     elderly_intervention,
                     household_intervention,
                     school_closer_intervention
                     ]
    return interventions


def scenario_262_interventions(compliance, ci_delay, hi_delay):
    ci_intervention = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        delay=ci_delay
    )
    sd_intervention = SocialDistancingIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        age_range=(0, 99)
    )
    periodic_school_closure_intervention = SchoolClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(60),
        compliance=1.0,
        proportion_of_envs=1.0,
        city_name='all',
        age_segment=(3, 9),
        period_data=AttributeAndPeriodData(
            EnvironmentalAttribute('household', 'last name', 2),
            timedelta(7)
        )
    )
    school_closer_intervention = SchoolClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(60),
        compliance=1.0,
        proportion_of_envs=0.98,
        city_name='all',
        age_segment=(10, 22)
    )
    workplace_closure_intervention = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(60),
        compliance=0.70
    )
    elderly_intervention = ElderlyQuarantineIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        min_age=65
    )
    household_intervention = HouseholdIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        delay_on_enter=hi_delay
    )
    interventions = [ci_intervention,
                     sd_intervention,
                     workplace_closure_intervention,
                     elderly_intervention,
                     household_intervention,
                     school_closer_intervention,
                     periodic_school_closure_intervention
                     ]
    return interventions


def scenario_272_interventions(compliance, ci_delay, hi_delay):
    ci_intervention = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        delay=ci_delay
    )
    sd_intervention = SocialDistancingIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        age_range=(0, 99)
    )
    periodic_school_closure_intervention = SchoolClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(60),
        compliance=1.0,
        proportion_of_envs=1.0,
        city_name='all',
        age_segment=(3, 6),
        period_data=AttributeAndPeriodData(
            EnvironmentalAttribute('household', 'last name', 2),
            timedelta(7)
        )
    )
    school_closer_intervention = SchoolClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(60),
        compliance=1.0,
        proportion_of_envs=0.98,
        city_name='all',
        age_segment=(7, 22)
    )
    workplace_closure_intervention = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(60),
        compliance=0.70
    )
    elderly_intervention = ElderlyQuarantineIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        min_age=65
    )
    household_intervention = HouseholdIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        delay_on_enter=hi_delay
    )
    interventions = [ci_intervention,
                     sd_intervention,
                     workplace_closure_intervention,
                     elderly_intervention,
                     household_intervention,
                     school_closer_intervention,
                     periodic_school_closure_intervention
                     ]
    return interventions


def scenario_282_interventions(compliance, ci_delay, hi_delay):
    ci_intervention = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        delay=ci_delay
    )
    sd_intervention = SocialDistancingIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        age_range=(0, 99)
    )
    school_closer_intervention = SchoolClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(60),
        compliance=1.0,
        proportion_of_envs=0.98,
        city_name='all',
        age_segment=(7, 22)
    )
    workplace_closure_intervention = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(60),
        compliance=0.70
    )
    elderly_intervention = ElderlyQuarantineIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        min_age=65
    )
    household_intervention = HouseholdIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        delay_on_enter=hi_delay
    )
    interventions = [ci_intervention,
                     sd_intervention,
                     workplace_closure_intervention,
                     elderly_intervention,
                     household_intervention,
                     school_closer_intervention
                     ]
    return interventions


# scenarios 36, 39, 365, 395
def scenario_36_interventions(compliance, ci_delay, hi_delay):
    ci_intervention = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        delay=ci_delay
    )
    sd_intervention = SocialDistancingIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        age_range=(0, 99)
    )
    school_closer_intervention = SchoolClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(60),
        compliance=1.0,
        proportion_of_envs=0.98,
        city_name='all',
        age_segment=(7, 22)
    )
    workplace_closure_intervention = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(60),
        compliance=0.70
    )
    elderly_intervention = ElderlyQuarantineIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        min_age=65
    )
    household_intervention = HouseholdIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        delay_on_enter=hi_delay
    )
    interventions = [ci_intervention,
                     sd_intervention,
                     workplace_closure_intervention,
                     elderly_intervention,
                     household_intervention,
                     school_closer_intervention
                     ]
    return interventions


def scenario_39_interventions(compliance, ci_delay, hi_delay):
    ci_intervention = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        delay=ci_delay
    )
    sd_intervention = SocialDistancingIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        age_range=(0, 99)
    )
    school_closer_intervention = SchoolClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(60),
        compliance=1.0,
        proportion_of_envs=0.98,
        city_name='all',
        age_segment=(10, 22)
    )
    workplace_closure_intervention = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(60),
        compliance=0.70
    )
    elderly_intervention = ElderlyQuarantineIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        min_age=65
    )
    household_intervention = HouseholdIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        delay_on_enter=hi_delay
    )
    interventions = [ci_intervention,
                     sd_intervention,
                     workplace_closure_intervention,
                     elderly_intervention,
                     household_intervention,
                     school_closer_intervention
                     ]
    return interventions


def scenario_365_interventions(compliance, ci_delay, hi_delay):
    ci_intervention = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        delay=ci_delay
    )
    sd_intervention = SocialDistancingIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        age_range=(0, 99)
    )
    periodic_school_closure_intervention = SchoolClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(60),
        compliance=1.0,
        proportion_of_envs=1.0,
        city_name='all',
        age_segment=(3, 6),
        period_data=AttributeAndPeriodData(
            EnvironmentalAttribute('household', 'last name', 2),
            timedelta(7)
        )
    )
    school_closer_intervention = SchoolClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(60),
        compliance=1.0,
        proportion_of_envs=0.98,
        city_name='all',
        age_segment=(7, 22)
    )
    workplace_closure_intervention = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(60),
        compliance=0.70
    )
    elderly_intervention = ElderlyQuarantineIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        min_age=65
    )
    household_intervention = HouseholdIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        delay_on_enter=hi_delay
    )
    interventions = [ci_intervention,
                     sd_intervention,
                     workplace_closure_intervention,
                     elderly_intervention,
                     household_intervention,
                     school_closer_intervention,
                     periodic_school_closure_intervention
                     ]
    return interventions


def scenario_395_interventions(compliance, ci_delay, hi_delay):
    ci_intervention = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        delay=ci_delay
    )
    sd_intervention = SocialDistancingIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        age_range=(0, 99)
    )
    periodic_school_closure_intervention = SchoolClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(60),
        compliance=1.0,
        proportion_of_envs=1.0,
        city_name='all',
        age_segment=(3, 9),
        period_data=AttributeAndPeriodData(
            EnvironmentalAttribute('household', 'last name', 2),
            timedelta(7)
        )
    )
    school_closer_intervention = SchoolClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(60),
        compliance=1.0,
        proportion_of_envs=0.98,
        city_name='all',
        age_segment=(10, 22)
    )
    workplace_closure_intervention = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(60),
        compliance=0.70
    )
    elderly_intervention = ElderlyQuarantineIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        min_age=65
    )
    household_intervention = HouseholdIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(60),
        compliance=compliance,
        delay_on_enter=hi_delay
    )
    interventions = [ci_intervention,
                     sd_intervention,
                     workplace_closure_intervention,
                     elderly_intervention,
                     household_intervention,
                     school_closer_intervention,
                     periodic_school_closure_intervention
                     ]
    return interventions


# trying to mimic reality for paper
def scenario_reality1(compliance, ci_delay, hi_delay):
    ci_intervention = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(90),
        compliance=compliance,
        delay=ci_delay
    )
    household_intervention = HouseholdIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(90),
        compliance=compliance,
        delay_on_enter=hi_delay
    )
    sd_intervention1 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(14),
        duration=timedelta(4),
        compliance=0.5,
        age_range=(0, 60)
    )
    sd_intervention2 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(18),
        duration=timedelta(22),
        compliance=0.7,
        age_range=(0, 60)
    )
    sd_intervention3 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(40),
        duration=timedelta(12),
        compliance=0.8,
        age_range=(0, 60)
    )
    sd_intervention4 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(52),
        duration=timedelta(10),
        compliance=0.5,
        age_range=(0, 60)
    )
    sd_intervention5 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(14),
        duration=timedelta(10),
        compliance=0.9,
        age_range=(61, 99)
    )
    school_closer_intervention1 = SchoolClosureIntervention(
        start_date=INITIAL_DATE + timedelta(15),
        duration=daysdelta(34),
        compliance=1.0,
        proportion_of_envs=1.0,
        city_name='all',
        age_segment=(3, 22)
    )
    school_closer_intervention2 = SchoolClosureIntervention(
        start_date=INITIAL_DATE + timedelta(49),
        duration=daysdelta(60),
        compliance=1.0,
        proportion_of_envs=0.5,
        city_name='all',
        age_segment=(3, 22)
    )
    workplace_closure_intervention = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE + timedelta(18),
        duration=daysdelta(30),
        compliance=0.70
    )

    interventions = [ci_intervention,
                     sd_intervention1,
                     sd_intervention2,
                     sd_intervention3,
                     sd_intervention4,
                     sd_intervention5,
                     workplace_closure_intervention,
                     household_intervention,
                     school_closer_intervention1,
                     school_closer_intervention2
                     ]
    return interventions


def scenario_check(compliance, ci_delay, hi_delay):
    ci_intervention = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(5),
        compliance=0.7,
        delay=0
    )
    household_intervention = HouseholdIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(90),
        compliance=compliance,
        delay_on_enter=hi_delay
    )
    sd_intervention1 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(14),
        duration=timedelta(4),
        compliance=0.9,
        age_range=(0, 99)
    )

    school_closer_intervention1 = SchoolClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(5),
        compliance=1.0,
        proportion_of_envs=1.0,
        city_name='all',
        age_segment=(3, 22)
    )
    school_closer_intervention2 = SchoolClosureIntervention(
        start_date=INITIAL_DATE + timedelta(6),
        duration=daysdelta(5),
        compliance=1.0,
        proportion_of_envs=1,
        city_name='all',
        age_segment=(3, 22)
    )
    school_closer_intervention3 = SchoolClosureIntervention(
        start_date=INITIAL_DATE + timedelta(12),
        duration=daysdelta(90),
        compliance=1.0,
        proportion_of_envs=1,
        city_name='all',
        age_segment=(3, 22)
    )
    workplace_closure_intervention = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE + timedelta(18),
        duration=daysdelta(30),
        compliance=0.70
    )

    interventions = [ci_intervention,
                     sd_intervention1,
                     workplace_closure_intervention,
                     household_intervention,
                     school_closer_intervention1,
                     school_closer_intervention2,
                     school_closer_intervention3
                     ]
    return interventions


def scenario_reality2(compliance, ci_delay, hi_delay):
    ci_intervention1 = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(14),
        compliance=0.3,
        delay=5
    )
    ci_intervention2 = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE + timedelta(14),
        duration=timedelta(14),
        compliance=0.45,
        delay=4
    )
    ci_intervention3 = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE + timedelta(28),
        duration=timedelta(21),
        compliance=0.6,
        delay=4
    )
    ci_intervention4 = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE + timedelta(49),
        duration=timedelta(100),
        compliance=0.75,
        delay=2
    )
    sd_intervention1 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(14),
        duration=timedelta(4),
        compliance=0.3,
        age_range=(0, 60)
    )
    sd_intervention2 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(18),
        duration=timedelta(2),
        compliance=0.45,
        age_range=(0, 60)
    )
    sd_intervention3 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(20),
        duration=timedelta(8),
        compliance=0.6,
        age_range=(0, 60)
    )
    sd_intervention4 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(28),
        duration=timedelta(12),
        compliance=0.7,
        age_range=(0, 60)
    )
    sd_intervention5 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(40),
        duration=timedelta(9),
        compliance=0.85,
        age_range=(0, 60)
    )
    sd_intervention6 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(49),
        duration=timedelta(100),
        compliance=0.7,
        age_range=(0, 60)
    )

    sd_intervention_eld_1 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(14),
        duration=timedelta(4),
        compliance=0.45,
        age_range=(61, 99)
    )
    sd_intervention_eld_2 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(18),
        duration=timedelta(2),
        compliance=0.6,
        age_range=(61, 99)
    )
    sd_intervention_eld_3 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(20),
        duration=timedelta(8),
        compliance=0.75,
        age_range=(61, 99)
    )
    sd_intervention_eld_4 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(28),
        duration=timedelta(12),
        compliance=0.85,
        age_range=(61, 99)
    )
    sd_intervention_eld_5 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(40),
        duration=timedelta(9),
        compliance=0.9,
        age_range=(61, 99)
    )
    sd_intervention_eld_6 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(49),
        duration=timedelta(100),
        compliance=0.85,
        age_range=(61, 99)
    )
    school_closer_intervention1 = SchoolClosureIntervention(
        start_date=INITIAL_DATE + timedelta(15),
        duration=daysdelta(34),
        compliance=1.0,
        proportion_of_envs=1.0,
        city_name='all',
        age_segment=(3, 22)
    )
    school_closer_intervention2 = SchoolClosureIntervention(
        start_date=INITIAL_DATE + timedelta(49),
        duration=daysdelta(100),
        compliance=1.0,
        proportion_of_envs=0.5,
        city_name='all',
        age_segment=(3, 22)
    )
    workplace_closure_intervention1 = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE + timedelta(18),
        duration=daysdelta(2),
        compliance=0.3
    )
    workplace_closure_intervention2 = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE + timedelta(20),
        duration=daysdelta(20),
        compliance=0.45
    )
    workplace_closure_intervention3 = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE + timedelta(40),
        duration=daysdelta(100),
        compliance=0.9
    )
    household_intervention = HouseholdIsolationIntervention(
        start_date=INITIAL_DATE + timedelta(49),
        duration=timedelta(100),
        compliance=0.75,
        delay_on_enter=2
    )

    interventions = [ci_intervention1,
                     ci_intervention2,
                     ci_intervention3,
                     ci_intervention4,
                     sd_intervention1,
                     sd_intervention2,
                     sd_intervention3,
                     sd_intervention4,
                     sd_intervention5,
                     sd_intervention6,
                     sd_intervention_eld_1,
                     sd_intervention_eld_2,
                     sd_intervention_eld_3,
                     sd_intervention_eld_4,
                     sd_intervention_eld_5,
                     sd_intervention_eld_6,
                     workplace_closure_intervention1,
                     workplace_closure_intervention2,
                     workplace_closure_intervention3,
                     household_intervention,
                     school_closer_intervention1,
                     school_closer_intervention2
                     ]
    return interventions


def scenario_reality3(compliance, ci_delay, hi_delay):
    ci_intervention1 = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(14),
        compliance=0.5,
        delay=5
    )
    ci_intervention2 = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE + timedelta(14),
        duration=timedelta(14),
        compliance=0.5,
        delay=4
    )
    ci_intervention3 = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE + timedelta(28),
        duration=timedelta(21),
        compliance=0.6,
        delay=4
    )
    ci_intervention4 = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE + timedelta(49),
        duration=timedelta(100),
        compliance=0.75,
        delay=2
    )
    sd_intervention1 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(14),
        duration=timedelta(4),
        compliance=0.2,
        age_range=(0, 60)
    )
    sd_intervention2 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(18),
        duration=timedelta(2),
        compliance=0.45,
        age_range=(0, 60)
    )
    sd_intervention3 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(20),
        duration=timedelta(8),
        compliance=0.6,
        age_range=(0, 60)
    )
    sd_intervention4 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(28),
        duration=timedelta(12),
        compliance=0.7,
        age_range=(0, 60)
    )
    sd_intervention5 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(40),
        duration=timedelta(9),
        compliance=0.85,
        age_range=(0, 60)
    )
    sd_intervention6 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(49),
        duration=timedelta(100),
        compliance=0.7,
        age_range=(0, 60)
    )

    sd_intervention_eld_1 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(14),
        duration=timedelta(4),
        compliance=0.55,
        age_range=(61, 99)
    )
    sd_intervention_eld_2 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(18),
        duration=timedelta(2),
        compliance=0.6,
        age_range=(61, 99)
    )
    sd_intervention_eld_3 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(20),
        duration=timedelta(8),
        compliance=0.75,
        age_range=(61, 99)
    )
    sd_intervention_eld_4 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(28),
        duration=timedelta(12),
        compliance=0.85,
        age_range=(61, 99)
    )
    sd_intervention_eld_5 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(40),
        duration=timedelta(9),
        compliance=0.9,
        age_range=(61, 99)
    )
    sd_intervention_eld_6 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(49),
        duration=timedelta(100),
        compliance=0.85,
        age_range=(61, 99)
    )
    school_closer_intervention1 = SchoolClosureIntervention(
        start_date=INITIAL_DATE + timedelta(15),
        duration=daysdelta(34),
        compliance=1.0,
        proportion_of_envs=1.0,
        city_name='all',
        age_segment=(3, 22)
    )
    school_closer_intervention2 = SchoolClosureIntervention(
        start_date=INITIAL_DATE + timedelta(49),
        duration=daysdelta(100),
        compliance=1.0,
        proportion_of_envs=0.5,
        city_name='all',
        age_segment=(3, 22)
    )
    workplace_closure_intervention1 = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE + timedelta(18),
        duration=daysdelta(2),
        compliance=0.3
    )
    workplace_closure_intervention2 = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE + timedelta(20),
        duration=daysdelta(20),
        compliance=0.45
    )
    workplace_closure_intervention3 = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE + timedelta(40),
        duration=daysdelta(100),
        compliance=0.9
    )
    household_intervention = HouseholdIsolationIntervention(
        start_date=INITIAL_DATE + timedelta(49),
        duration=timedelta(100),
        compliance=0.75,
        delay_on_enter=2
    )

    interventions = [ci_intervention1,
                     ci_intervention2,
                     ci_intervention3,
                     ci_intervention4,
                     sd_intervention1,
                     sd_intervention2,
                     sd_intervention3,
                     sd_intervention4,
                     sd_intervention5,
                     sd_intervention6,
                     sd_intervention_eld_1,
                     sd_intervention_eld_2,
                     sd_intervention_eld_3,
                     sd_intervention_eld_4,
                     sd_intervention_eld_5,
                     sd_intervention_eld_6,
                     workplace_closure_intervention1,
                     workplace_closure_intervention2,
                     workplace_closure_intervention3,
                     household_intervention,
                     school_closer_intervention1,
                     school_closer_intervention2
                     ]
    return interventions


def scenario_reality4(compliance, ci_delay, hi_delay):
    ci_intervention1 = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(14),
        compliance=0.3,
        delay=5
    )
    ci_intervention2 = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE + timedelta(14),
        duration=timedelta(14),
        compliance=0.45,
        delay=4
    )
    ci_intervention3 = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE + timedelta(28),
        duration=timedelta(21),
        compliance=0.8,
        delay=4
    )
    ci_intervention4 = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE + timedelta(49),
        duration=timedelta(100),
        compliance=0.85,
        delay=2
    )
    sd_intervention1 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(14),
        duration=timedelta(4),
        compliance=0.3,
        age_range=(0, 60)
    )
    sd_intervention2 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(18),
        duration=timedelta(2),
        compliance=0.45,
        age_range=(0, 60)
    )
    sd_intervention3 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(20),
        duration=timedelta(8),
        compliance=0.6,
        age_range=(0, 60)
    )
    sd_intervention4 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(28),
        duration=timedelta(12),
        compliance=0.7,
        age_range=(0, 60)
    )
    sd_intervention5 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(40),
        duration=timedelta(9),
        compliance=0.85,
        age_range=(0, 60)
    )
    sd_intervention6 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(49),
        duration=timedelta(100),
        compliance=0.7,
        age_range=(0, 60)
    )

    sd_intervention_eld_1 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(14),
        duration=timedelta(4),
        compliance=0.45,
        age_range=(61, 99)
    )
    sd_intervention_eld_2 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(18),
        duration=timedelta(2),
        compliance=0.6,
        age_range=(61, 99)
    )
    sd_intervention_eld_3 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(20),
        duration=timedelta(8),
        compliance=0.75,
        age_range=(61, 99)
    )
    sd_intervention_eld_4 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(28),
        duration=timedelta(12),
        compliance=0.85,
        age_range=(61, 99)
    )
    sd_intervention_eld_5 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(40),
        duration=timedelta(9),
        compliance=0.9,
        age_range=(61, 99)
    )
    sd_intervention_eld_6 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(49),
        duration=timedelta(100),
        compliance=0.85,
        age_range=(61, 99)
    )
    school_closer_intervention1 = SchoolClosureIntervention(
        start_date=INITIAL_DATE + timedelta(15),
        duration=daysdelta(34),
        compliance=1.0,
        proportion_of_envs=1.0,
        city_name='all',
        age_segment=(3, 22)
    )
    school_closer_intervention2 = SchoolClosureIntervention(
        start_date=INITIAL_DATE + timedelta(49),
        duration=daysdelta(100),
        compliance=1.0,
        proportion_of_envs=0.5,
        city_name='all',
        age_segment=(3, 22)
    )
    workplace_closure_intervention1 = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE + timedelta(18),
        duration=daysdelta(2),
        compliance=0.3
    )
    workplace_closure_intervention2 = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE + timedelta(20),
        duration=daysdelta(20),
        compliance=0.45
    )
    workplace_closure_intervention3 = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE + timedelta(40),
        duration=daysdelta(100),
        compliance=0.9
    )
    household_intervention = HouseholdIsolationIntervention(
        start_date=INITIAL_DATE + timedelta(49),
        duration=timedelta(100),
        compliance=0.85,
        delay_on_enter=2
    )

    interventions = [ci_intervention1,
                     ci_intervention2,
                     ci_intervention3,
                     ci_intervention4,
                     sd_intervention1,
                     sd_intervention2,
                     sd_intervention3,
                     sd_intervention4,
                     sd_intervention5,
                     sd_intervention6,
                     sd_intervention_eld_1,
                     sd_intervention_eld_2,
                     sd_intervention_eld_3,
                     sd_intervention_eld_4,
                     sd_intervention_eld_5,
                     sd_intervention_eld_6,
                     workplace_closure_intervention1,
                     workplace_closure_intervention2,
                     workplace_closure_intervention3,
                     household_intervention,
                     school_closer_intervention1,
                     school_closer_intervention2
                     ]
    return interventions


def no_interventions(compliance, ci_delay, hi_delay):
    interventions = []
    return interventions


def not_relaxing_interventions(compliance, ci_delay, hi_delay):
    ci_intervention1 = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(14),
        compliance=0.3,
        delay=5
    )
    ci_intervention2 = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE + timedelta(14),
        duration=timedelta(14),
        compliance=0.45,
        delay=4
    )
    ci_intervention3 = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE + timedelta(28),
        duration=timedelta(21),
        compliance=0.6,
        delay=4
    )
    ci_intervention4 = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE + timedelta(49),
        duration=timedelta(100),
        compliance=0.75,
        delay=2
    )
    sd_intervention1 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(14),
        duration=timedelta(4),
        compliance=0.3,
        age_range=(0, 60)
    )
    sd_intervention2 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(18),
        duration=timedelta(2),
        compliance=0.45,
        age_range=(0, 60)
    )
    sd_intervention3 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(20),
        duration=timedelta(8),
        compliance=0.6,
        age_range=(0, 60)
    )
    sd_intervention4 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(28),
        duration=timedelta(12),
        compliance=0.7,
        age_range=(0, 60)
    )
    sd_intervention5 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(40),
        duration=timedelta(100),
        compliance=0.85,
        age_range=(0, 60)
    )

    sd_intervention_eld_1 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(14),
        duration=timedelta(4),
        compliance=0.45,
        age_range=(61, 99)
    )
    sd_intervention_eld_2 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(18),
        duration=timedelta(2),
        compliance=0.6,
        age_range=(61, 99)
    )
    sd_intervention_eld_3 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(20),
        duration=timedelta(8),
        compliance=0.75,
        age_range=(61, 99)
    )
    sd_intervention_eld_4 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(28),
        duration=timedelta(12),
        compliance=0.85,
        age_range=(61, 99)
    )
    sd_intervention_eld_5 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(40),
        duration=timedelta(100),
        compliance=0.9,
        age_range=(61, 99)
    )

    school_closer_intervention1 = SchoolClosureIntervention(
        start_date=INITIAL_DATE + timedelta(15),
        duration=daysdelta(100),
        compliance=1.0,
        proportion_of_envs=1.0,
        city_name='all',
        age_segment=(3, 22)
    )

    workplace_closure_intervention1 = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE + timedelta(18),
        duration=daysdelta(2),
        compliance=0.3
    )
    workplace_closure_intervention2 = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE + timedelta(20),
        duration=daysdelta(20),
        compliance=0.45
    )
    workplace_closure_intervention3 = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE + timedelta(40),
        duration=daysdelta(100),
        compliance=0.9
    )
    household_intervention = HouseholdIsolationIntervention(
        start_date=INITIAL_DATE + timedelta(49),
        duration=timedelta(100),
        compliance=0.75,
        delay_on_enter=2
    )

    interventions = [ci_intervention1,
                     ci_intervention2,
                     ci_intervention3,
                     ci_intervention4,
                     sd_intervention1,
                     sd_intervention2,
                     sd_intervention3,
                     sd_intervention4,
                     sd_intervention5,
                     sd_intervention_eld_1,
                     sd_intervention_eld_2,
                     sd_intervention_eld_3,
                     sd_intervention_eld_4,
                     sd_intervention_eld_5,
                     workplace_closure_intervention1,
                     workplace_closure_intervention2,
                     workplace_closure_intervention3,
                     household_intervention,
                     school_closer_intervention1
                     ]
    return interventions


# for grant - july 2020
def grant_time1(compliance, ci_delay, hi_delay):
    ci_intervention1 = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE + timedelta(21),
        duration=timedelta(10),
        compliance=0.3,
        delay=5
    )
    ci_intervention2 = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE + timedelta(31),
        duration=timedelta(43),
        compliance=0.9,
        delay=4
    )
    sd_intervention1 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(21),
        duration=timedelta(53),
        compliance=0.8,
        age_range=(0, 60)
    )
    sd_intervention_eld_1 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(21),
        duration=timedelta(53),
        compliance=0.9,
        age_range=(61, 99)
    )

    school_closer_intervention1 = SchoolClosureIntervention(
        start_date=INITIAL_DATE + timedelta(21),
        duration=daysdelta(53),
        compliance=1.0,
        proportion_of_envs=1.0,
        city_name='all',
        age_segment=(3, 22)
    )
    workplace_closure_intervention1 = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE + timedelta(21),
        duration=daysdelta(53),
        compliance=0.7
    )

    household_intervention = HouseholdIsolationIntervention(
        start_date=INITIAL_DATE + timedelta(21),
        duration=timedelta(53),
        compliance=0.7,
        delay_on_enter=2
    )

    ci_intervention0 = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE + timedelta(120),
        duration=timedelta(60),
        compliance=0.8,
        delay=4
    )
    sd_intervention0 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(120),
        duration=timedelta(60),
        compliance=0.6,
        age_range=(0, 60)
    )
    sd_intervention_eld_0 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(120),
        duration=timedelta(60),
        compliance=0.8,
        age_range=(61, 99)
    )
    workplace_closure_intervention0 = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE + timedelta(120),
        duration=daysdelta(60),
        compliance=0.7
    )

    household_intervention0 = HouseholdIsolationIntervention(
        start_date=INITIAL_DATE + timedelta(120),
        duration=timedelta(60),
        compliance=0.7,
        delay_on_enter=2
    )

    interventions = [ci_intervention1,
                     ci_intervention2,
                     sd_intervention1,
                     sd_intervention_eld_1,
                     workplace_closure_intervention1,
                     household_intervention,
                     school_closer_intervention1
                     ]
    return interventions


def grant_time2(compliance, ci_delay, hi_delay):
    ci_intervention1 = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE + timedelta(14),
        duration=timedelta(10),
        compliance=0.3,
        delay=5
    )
    ci_intervention2 = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE + timedelta(24),
        duration=timedelta(64),
        compliance=0.9,
        delay=4
    )
    sd_intervention1 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(14),
        duration=timedelta(74),
        compliance=0.8,
        age_range=(0, 60)
    )
    sd_intervention_eld_1 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(14),
        duration=timedelta(74),
        compliance=0.9,
        age_range=(61, 99)
    )

    school_closer_intervention1 = SchoolClosureIntervention(
        start_date=INITIAL_DATE + timedelta(14),
        duration=daysdelta(60),
        compliance=1.0,
        proportion_of_envs=1.0,
        city_name='all',
        age_segment=(3, 22)
    )
    workplace_closure_intervention1 = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE + timedelta(14),
        duration=daysdelta(74),
        compliance=0.7
    )

    household_intervention = HouseholdIsolationIntervention(
        start_date=INITIAL_DATE + timedelta(14),
        duration=timedelta(74),
        compliance=0.7,
        delay_on_enter=2
    )
    ci_intervention0 = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE + timedelta(120),
        duration=timedelta(60),
        compliance=0.8,
        delay=4
    )
    sd_intervention0 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(120),
        duration=timedelta(60),
        compliance=0.6,
        age_range=(0, 60)
    )
    sd_intervention_eld_0 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(120),
        duration=timedelta(60),
        compliance=0.8,
        age_range=(61, 99)
    )
    workplace_closure_intervention0 = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE + timedelta(120),
        duration=daysdelta(60),
        compliance=0.7
    )

    household_intervention0 = HouseholdIsolationIntervention(
        start_date=INITIAL_DATE + timedelta(120),
        duration=timedelta(60),
        compliance=0.7,
        delay_on_enter=2
    )

    interventions = [ci_intervention1,
                     ci_intervention2,
                     sd_intervention1,
                     sd_intervention_eld_1,
                     workplace_closure_intervention1,
                     household_intervention,
                     school_closer_intervention1
                     ]
    return interventions


# for bene beraq paper
def paper_1(compliance, ci_delay, hi_delay):
    interventions = []
    return interventions


def no_interventions(compliance=1, ci_delay=None, hi_delay=None):
    return []


def vaccinations_scenario_general(compliance=1, ci_delay=None, hi_delay=None):
    vaccinations = ImmuneGeneralPopulationIntervention(
        compliance=compliance,
        start_date=INITIAL_DATE,
        duration=daysdelta(16 * 7),
        people_per_day=100,
        min_age=18)
    interventions = [
        vaccinations
    ]
    return interventions


def paper_2(compliance, ci_delay, hi_delay):
    interention_start_delay = timedelta(days=21)
    ci_intervention = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE + interention_start_delay,
        duration=timedelta(180),
        compliance=0.8,
        delay=4
    )

    sd_intervention = SocialDistancingIntervention(
        start_date=INITIAL_DATE + interention_start_delay,
        duration=timedelta(180),
        compliance=0.8,
        age_range=(0, 99)
    )

    school_closer_intervention = SchoolClosureIntervention(
        start_date=INITIAL_DATE + interention_start_delay,
        duration=daysdelta(180),
        compliance=1.0,
        proportion_of_envs=1.0,
        city_name='all',
        age_segment=(3, 22)
    )

    workplace_closure_intervention = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE + interention_start_delay,
        duration=daysdelta(180),
        compliance=1.0
    )

    household_intervention = HouseholdIsolationIntervention(
        start_date=INITIAL_DATE + interention_start_delay,
        duration=timedelta(180),
        compliance=0.8,
        delay_on_enter=4
    )

    interventions = [ci_intervention,
                     sd_intervention,
                     workplace_closure_intervention,
                     household_intervention,
                     school_closer_intervention
                     ]
    return interventions


def paper_3(compliance, ci_delay, hi_delay):
    ci_intervention = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(14),
        compliance=0.8,
        delay=4
    )

    sd_intervention = SocialDistancingIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(14),
        compliance=0.8,
        age_range=(0, 99)
    )

    school_closer_intervention = SchoolClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(14),
        compliance=1.0,
        proportion_of_envs=1.0,
        city_name='all',
        age_segment=(3, 22)
    )

    workplace_closure_intervention = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(14),
        compliance=1.0
    )

    household_intervention = HouseholdIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(14),
        compliance=0.8,
        delay_on_enter=4
    )

    interventions = [ci_intervention,
                     sd_intervention,
                     workplace_closure_intervention,
                     household_intervention,
                     school_closer_intervention
                     ]
    return interventions


def paper_4(compliance, ci_delay, hi_delay):
    ci_intervention = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(28),
        compliance=0.8,
        delay=4
    )

    sd_intervention = SocialDistancingIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(28),
        compliance=0.8,
        age_range=(0, 99)
    )

    school_closer_intervention = SchoolClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(28),
        compliance=1.0,
        proportion_of_envs=1.0,
        city_name='all',
        age_segment=(3, 22)
    )

    workplace_closure_intervention = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(28),
        compliance=1.0
    )

    household_intervention = HouseholdIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(28),
        compliance=0.8,
        delay_on_enter=4
    )

    interventions = [ci_intervention,
                     sd_intervention,
                     workplace_closure_intervention,
                     household_intervention,
                     school_closer_intervention
                     ]
    return interventions


def paper_5(compliance, ci_delay, hi_delay):
    ci_intervention = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(56),
        compliance=0.8,
        delay=4
    )

    sd_intervention = SocialDistancingIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(56),
        compliance=0.8,
        age_range=(0, 99)
    )

    school_closer_intervention = SchoolClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(14),
        compliance=1.0,
        proportion_of_envs=1.0,
        city_name='all',
        age_segment=(3, 22)
    )

    workplace_closure_intervention = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(56),
        compliance=1.0
    )

    household_intervention = HouseholdIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(56),
        compliance=0.8,
        delay_on_enter=4
    )

    interventions = [ci_intervention,
                     sd_intervention,
                     workplace_closure_intervention,
                     household_intervention,
                     school_closer_intervention
                     ]
    return interventions


def paper_6(compliance, ci_delay, hi_delay):
    ci_intervention = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(56),
        compliance=0.8,
        delay=4
    )

    sd_intervention = SocialDistancingIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(56),
        compliance=0.8,
        age_range=(0, 99)
    )

    school_closer_intervention = SchoolClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(56),
        compliance=1.0,
        proportion_of_envs=1.0,
        city_name='all',
        age_segment=(3, 22)
    )

    workplace_closure_intervention = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(14),
        compliance=1.0
    )

    household_intervention = HouseholdIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(56),
        compliance=0.8,
        delay_on_enter=4
    )

    interventions = [ci_intervention,
                     sd_intervention,
                     workplace_closure_intervention,
                     household_intervention,
                     school_closer_intervention
                     ]
    return interventions


def paper_7(compliance, ci_delay, hi_delay):
    ci_intervention = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(56),
        compliance=0.8,
        delay=4
    )

    sd_intervention = SocialDistancingIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(56),
        compliance=0.8,
        age_range=(0, 99)
    )

    school_closer_intervention = SchoolClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(14),
        compliance=1.0,
        proportion_of_envs=1.0,
        city_name='all',
        age_segment=(3, 22)
    )

    workplace_closure_intervention = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(14),
        compliance=1.0
    )

    household_intervention = HouseholdIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(56),
        compliance=0.8,
        delay_on_enter=4
    )

    interventions = [ci_intervention,
                     sd_intervention,
                     workplace_closure_intervention,
                     household_intervention,
                     school_closer_intervention
                     ]
    return interventions


def paper_8(compliance, ci_delay, hi_delay):
    ci_intervention1 = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(14),
        compliance=0.3,
        delay=5
    )
    ci_intervention2 = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE + timedelta(14),
        duration=timedelta(14),
        compliance=0.45,
        delay=4
    )
    ci_intervention3 = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE + timedelta(28),
        duration=timedelta(21),
        compliance=0.6,
        delay=4
    )
    ci_intervention4 = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE + timedelta(49),
        duration=timedelta(180),
        compliance=0.75,
        delay=2
    )
    sd_intervention1 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(14),
        duration=timedelta(4),
        compliance=0.3,
        age_range=(0, 60)
    )
    sd_intervention2 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(18),
        duration=timedelta(2),
        compliance=0.45,
        age_range=(0, 60)
    )
    sd_intervention3 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(20),
        duration=timedelta(8),
        compliance=0.6,
        age_range=(0, 60)
    )
    sd_intervention4 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(28),
        duration=timedelta(12),
        compliance=0.7,
        age_range=(0, 60)
    )
    sd_intervention5 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(40),
        duration=timedelta(9),
        compliance=0.85,
        age_range=(0, 60)
    )
    sd_intervention6 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(49),
        duration=timedelta(180),
        compliance=0.7,
        age_range=(0, 60)
    )

    sd_intervention_eld_1 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(14),
        duration=timedelta(4),
        compliance=0.45,
        age_range=(61, 99)
    )
    sd_intervention_eld_2 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(18),
        duration=timedelta(2),
        compliance=0.6,
        age_range=(61, 99)
    )
    sd_intervention_eld_3 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(20),
        duration=timedelta(8),
        compliance=0.75,
        age_range=(61, 99)
    )
    sd_intervention_eld_4 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(28),
        duration=timedelta(12),
        compliance=0.85,
        age_range=(61, 99)
    )
    sd_intervention_eld_5 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(40),
        duration=timedelta(9),
        compliance=0.9,
        age_range=(61, 99)
    )
    sd_intervention_eld_6 = SocialDistancingIntervention(
        start_date=INITIAL_DATE + timedelta(49),
        duration=timedelta(180),
        compliance=0.85,
        age_range=(61, 99)
    )
    school_closer_intervention1 = SchoolClosureIntervention(
        start_date=INITIAL_DATE + timedelta(15),
        duration=daysdelta(48),
        compliance=1.0,
        proportion_of_envs=1.0,
        city_name='all',
        age_segment=(3, 22)
    )
    school_closer_intervention2 = SchoolClosureIntervention(
        start_date=INITIAL_DATE + timedelta(63),
        duration=daysdelta(60),
        compliance=1.0,
        proportion_of_envs=0.5,
        city_name='all',
        age_segment=(3, 22)
    )
    workplace_closure_intervention1 = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE + timedelta(18),
        duration=daysdelta(2),
        compliance=0.3
    )
    workplace_closure_intervention2 = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE + timedelta(20),
        duration=daysdelta(20),
        compliance=0.45
    )
    workplace_closure_intervention3 = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE + timedelta(40),
        duration=daysdelta(10),
        compliance=0.9
    )
    workplace_closure_intervention4 = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE + timedelta(50),
        duration=daysdelta(180),
        compliance=0.6
    )
    household_intervention = HouseholdIsolationIntervention(
        start_date=INITIAL_DATE + timedelta(49),
        duration=timedelta(180),
        compliance=0.75,
        delay_on_enter=2
    )

    interventions = [ci_intervention1,
                     ci_intervention2,
                     ci_intervention3,
                     ci_intervention4,
                     sd_intervention1,
                     sd_intervention2,
                     sd_intervention3,
                     sd_intervention4,
                     sd_intervention5,
                     sd_intervention6,
                     sd_intervention_eld_1,
                     sd_intervention_eld_2,
                     sd_intervention_eld_3,
                     sd_intervention_eld_4,
                     sd_intervention_eld_5,
                     sd_intervention_eld_6,
                     workplace_closure_intervention1,
                     workplace_closure_intervention2,
                     workplace_closure_intervention3,
                     workplace_closure_intervention4,
                     household_intervention,
                     school_closer_intervention1,
                     school_closer_intervention2
                     ]
    return interventions


def paper_2_comp_9(compliance, ci_delay, hi_delay):
    ci_intervention = SymptomaticIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(180),
        compliance=0.9,
        delay=4
    )

    sd_intervention = SocialDistancingIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(180),
        compliance=0.9,
        age_range=(0, 99)
    )

    school_closer_intervention = SchoolClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(180),
        compliance=1.0,
        proportion_of_envs=1.0,
        city_name='all',
        age_segment=(3, 22)
    )

    workplace_closure_intervention = WorkplaceClosureIntervention(
        start_date=INITIAL_DATE,
        duration=daysdelta(180),
        compliance=1.0
    )

    household_intervention = HouseholdIsolationIntervention(
        start_date=INITIAL_DATE,
        duration=timedelta(180),
        compliance=0.9,
        delay_on_enter=4
    )

    interventions = [ci_intervention,
                     sd_intervention,
                     workplace_closure_intervention,
                     household_intervention,
                     school_closer_intervention
                     ]
    return interventions
