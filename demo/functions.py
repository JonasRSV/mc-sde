import numpy as np

def split_drift(t, x):
    return np.float_power(x, 3) + np.sign(x) * (1 / (np.abs(x) + 2))


def split_diffusion(t, x):
    return 0.1



def drift(t, x):
    return 0

def diffusion(t, x):
    return 1




# population sz is size of entire population
# probability of transmission is p("healty meet sick" -> healthy gets sick")
# rate of interaction is the number of individuals each individual meets each day

population_sz = int(1e7)
probability_of_transmission = 0.01
rate_of_interaction = 13 # How many people each individual meets per day
rate_of_recovery = 30 # How many days to recover 

has_been_sick = None
def covid_drift(t, x):
    global has_been_sick
    if has_been_sick is None:
        has_been_sick = np.zeros_like(x, dtype=np.float64)
    
    recovering = x * (1.0 / rate_of_recovery)
    has_been_sick = has_been_sick + recovering * 1.0
    
    return -recovering
    

def covid_diffusion(t, x):
    global has_been_sick
    if has_been_sick is None:
        has_been_sick = np.zeros_like(x)
        
    p_sick = x / population_sz    
    p_healthy = 1 - p_sick
    p_person_meet_sick = 1 - np.float_power(p_healthy, rate_of_interaction)
    p_person_not_immune = 1 - (has_been_sick / population_sz)
    
    p_person_get_sick = p_person_meet_sick * probability_of_transmission * p_person_not_immune
    
    expected_infections_per_time_unit = p_person_get_sick * np.maximum(population_sz - x, 0)

    return expected_infections_per_time_unit
 