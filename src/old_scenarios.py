from datetime import daysdelta, timedelta

from src.simulation.interventions import *
from src.world.environments import EnvironmentalAttribute
from src.run_utils import INITIAL_DATE
# examples of initializing different interventions

# case isolation - isolating to quarantine mode every person that develops symptoms
ci_intervention = SymptomaticIsolationIntervention(
    start_date=INITIAL_DATE,
    duration=daysdelta(500),
    compliance=0.7,
    delay=0
)



# house isolation - isolate the entire house when one of them develops symptoms
# here it's configures for 14 days from symptoms
hq_intervention = HouseholdIsolationIntervention(
    start_date=INITIAL_DATE + daysdelta(15),
    duration=daysdelta(120),
    compliance=0.7,
    delay_on_enter=0,
    delay_on_exit=14,
    is_exit_after_recovery=False
)

# social distancing - decrease contacts outside of household (mainly) for the given duration
sd_intervention = SocialDistancingIntervention(
    start_date=INITIAL_DATE + daysdelta(15),
    duration=daysdelta(120),
    compliance=0.75,
    age_range=(5, 10)
)

# school isolation - simulates kids staying at school and not returning
# here, it's configured for the entire country
school_isolation_intervention = SchoolIsolationIntervention(
    start_date=INITIAL_DATE + daysdelta(0),
    duration=daysdelta(120),
    compliance=0.7,
    proportion_of_envs=0.35,
    city_name='all',
    age_segment=(5, 12)
)

# workplace closure - close all working places
workplace_closure_intervention = WorkplaceClosureIntervention(
    start_date=INITIAL_DATE + daysdelta(10),
    duration=daysdelta(120),
    compliance=1.0
)

# distance all the elder people, here it's configured from age 70 and up
elderly_intervention = ElderlyQuarantineIntervention(
    start_date=INITIAL_DATE + daysdelta(30),
    duration=daysdelta(120),
    compliance=0.75,
    min_age=70
)

# simulates the closure of Jerusalem - no one is getting in or out for the duration that is given
jerusalem_intervention = CityCurfewIntervention(
    'jerusalem',
    start_date=INITIAL_DATE,
    duration=daysdelta(120),
    compliance=0.7
)

# simulates the closure of Tel Aviv - no one is getting in or out for the duration that is given
tlv_intervention = CityCurfewIntervention(
    'tel aviv-yafo',
    start_date=INITIAL_DATE,
    duration=daysdelta(120),
    compliance=0.7
)

# school closer - the kids in the given age range do not go to school
school_closure_intervention = SchoolClosureIntervention(
    start_date=INITIAL_DATE,
    duration=daysdelta(120),
    compliance=1.0,
    proportion_of_envs=1.0,
    city_name='all',
    age_segment=(3, 18)
)

# simulates a two week cycle in which every week, one half of the kids (by household) go to school
periodic_school_closure_intervention = SchoolClosureIntervention(
    start_date=INITIAL_DATE,
    duration=daysdelta(120),
    compliance=1.0,
    proportion_of_envs=1.0,
    city_name='all',
    age_segment=(3, 18),
    period_data=AttributeAndPeriodData(
        EnvironmentalAttribute('household', 'last name', 2),
        timedelta(7)
    )
)

# simulates a three week cycle in which every week, one third of the kids (by household) go to school
periodic_school_closure_intervention_2 = SchoolClosureIntervention(
    start_date=INITIAL_DATE,
    duration=daysdelta(120),
    compliance=1.0,
    proportion_of_envs=1.0,
    city_name='all',
    age_segment=(3, 18),
    period_data=AttributeAndPeriodData(
        EnvironmentalAttribute('household', 'last name', 3),
        timedelta(7)
    )
)

# examples of different scenarios of intervention together

# only case isolation
ci = [ci_intervention]

# case isolation and school closure
ci_pc = [ci_intervention, sd_intervention]

# case isolation ans workplace closure
ci_wc = [
    ci_intervention,
    workplace_closure_intervention
]

# Case Isolation and Household Quarantine
ci_hq = [ci_intervention, hq_intervention]

# School closing
# pc = [school_closure_intervention]

# CI, HQ, Elderly Quarantine
ci_sde = [
    ci_intervention,
    hq_intervention,
    elderly_intervention,
]

# CI, HQ, SD for entire population
ci_hq_sd = [
    ci_intervention,
    hq_intervention,
    sd_intervention
]

# Closing Schools, CI, SD for entire population
ci_pc_sd = [
    ci_intervention,
    sd_intervention,
    school_closure_intervention
]

# closing schools periodically (two weeks cycle), social distancing and case isolation
ci_sd_psc = [
    ci_intervention,
    sd_intervention,
    periodic_school_closure_intervention
]

# closing schools periodically (three weeks cycle), social distancing and case isolation
ci_sd_psc2 = [
    ci_intervention,
    sd_intervention,
    periodic_school_closure_intervention_2
]

# closing schools, isolating kids at school, social distancing and case isolation
ci_pc_pi_sd = [
    ci_intervention,
    sd_intervention,
    school_closure_intervention,
    school_isolation_intervention
]

# closing schools, isolating kids at school, work closure, social distancing and case isolation
ci_pc_pi_sd_wc = [
    ci_intervention,
    sd_intervention,
    school_closure_intervention,
    school_isolation_intervention,
    workplace_closure_intervention
]

# closure of Tel Aviv and Jerusalen with case isolation
city_cerfew = [
    ci_intervention,
    jerusalem_intervention,
    tlv_intervention
]


# all interventions to activate, all the uncommented line will be added as a different simulation run
# the keys are the names of the run, and the values are the list of active interventions in the run
interventions_modes = {
    'default': [],
    'CI': ci,
    # 'CI_SDE': ci_sde,
    'CI_PC': ci_pc,
    'CI_PC_SD': ci_pc_sd,
    'CI_PC_HSD': ci_sd_psc,
    'CI_PC_HSD2': ci_sd_psc2
    # 'CI_PC_PI_SD': ci_pc_pi_sd,
    # 'CI_WC': ci_wc,
    # 'CI_PC_PI_SD_WC': ci_pc_pi_sd_wc
    # 'CI': ci,
    # 'CI_SDE': ci_sde,
    # 'CI_PC_SD': ci_pc_sd,
    # 'CI_PC_PI_SD_WC': ci_pc_pi_sd_wc,
    # 'CC': city_cerfew
    # 'CI_HQ': ci_hq,
    # 'PC': pc,
    # 'CI_HQ_SD': ci_hq_sd,
}