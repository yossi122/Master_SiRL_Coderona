# This code takes demographic distributions and simulates typical households.
# Currently the code gets the age distribution of the population,
# the distribution of the households sizes, percentage of households with
# someone over 65, percentage of households with someone under 18,
# number of households, total population, percentage of households with
# children that have a single parent and percentage of children that have a
# single parent

# This data does not give a distribution on the types of households, so here
# are some assumptions I have made. Some of them make sense, some of them do
# not, but I had no other options without data. Some of them are easy to change
# in the code, and some will be harder

# Dependancy between over 65 and others. Do households with over 65 tend to
# have children, to have adults under 65? How many people over 65?
# Currently I decided that if there is someone over 65 there are 1 or 2 of them.
# Given we know how many people are over 65, we can calculate how many
# households have 1 and how many 2. As to dependancy, I chose to assume
# independance. If we will have better stats, it shouldn't be too hard to put
# them in the code

# Distributions of households above largest size (current data has percentage
# of households with 7+). I chose to simply distibute that percentage over the
# few sizes (with the last one less probable) so that number of households
# compared to total population will be correct. Here too there is one routine
# that does this at the beginning, so should be easy to fix if we have data or
# a better strategy

# Size of single parent housholds. I chose to assume that these households have
# 1 or 2 children, and calculate percentages so that the two stats work out

# The code works as follows:
# First we get all the distributions we need - how many people of each age,
# how many households with children, how many households of each size, how many
# households with 0 1 or 2 people over 65, how many households with single parent
# with 1 or 2 children.

# We now keep track of all these numbers, and every time we generate a household
# we subtract from the appropriate numbers, so totals will work out in the
# end. We need to make sure we won't run our of houses, or children, or people
# over 65 so we generate them in this order
# 1. Generate all households with single parent and single child. Lotter whether
#    there are over 65 people according to current stats. The adults and children
#    are also lottered given remaining numbers
# 2. Generate all households with single parent and two children. Same as above.
#    We choose these first because we know children won't run out in this stage
# 3. Generate all households with children. We assume all household with 6 or above
#    have children, so we lotter those first - given size lotter how many over 65
#    and then choose children according to how many are left to fill. Here too
#    we choose all population ages from number remaining. It is important
#    to notice we can run out of children or households, so once there aren't
#    enough children or households we choose minimal or maximal size households
#    respectively.
# 4. Generate all households with two people over 65 - choose total size from
#    remaining sizes (not very reasonable, and we can cprrect later if we
#    have stats)
# 5. Generate remaining households - randomly choose if there is 1 person over 65
#    and then choose size of households (again, we might want to change if we have
#    data)

import random
from bisect import bisect as _bisect



def sample_run():
    jer_pop = [[114857, 0, 5], [102064, 5, 10], [91524, 10, 15], [87179, 15, 20], [81406, 20, 25], [70012, 25, 30], [58684, 30, 35], [50458, 35, 40], [43938, 40, 45], [39029, 45, 50], [34179, 50, 55], [32449, 55, 60], [29286, 60, 65], [26258, 65, 70], [21397, 70, 75], [36718, 75, 95]]
    jer_nh_dist = [0, 18.1, 21.7, 13.9, 13.6, 11.6, 8.7, 12.4]
    jer_p65 = .218
    jer_pc = .505
    jer_nh = 202200
    jer_tot = 730000
    a = sim_houses(jer_pop, jer_nh_dist, jer_p65, jer_pc, int(jer_nh/10), int(jer_tot/10), 0.11, 0.09)
    return(a)

# Choose an index from a vector of sizes (we normalize to probabilities
# in the routine)
def rand_distrib(prbs, sums=None):
    if sums is None:
        s = sum(prbs)
        v = random.random()*(0.0+s)

        v0 = 0
        for i in range(len(prbs)):
            v0 += prbs[i]
            if v0 >= v:
                return(i)
        return(len(prbs)-1)
    assert len(prbs) + 1 == len(sums)
    return _bisect(sums, random.random() * sums[-1]) - 1

# Given a probability vector (we normalize) and a population, generate whole
# numbers that are close to probabilities and that sum to the population size
def from_probs_to_nums(prbs, n, needed_weighted_sum = None):
    s = sum(prbs)+0.0
    res = [int(prbs[i]*n/s) for i in range(len(prbs))]
    rems = [(res[i]-prbs[i]*n/s,i) for i in range(len(prbs))]
    rems.sort()
    
    n0 = sum(res)
    if (needed_weighted_sum is not None):
        cur_sum = sum([i*res[i] for i in range(len(res))])
        [res, v] = update_rems_from_weights(res, rems, needed_weighted_sum-cur_sum, n-n0)
        if (v==1):
            return(res)
    
    for i in range(n-n0):
        ind = rems[i][1]
        res[ind] += 1
    return(res)

# Use backtracking to get probable inds with correct sum
def update_rems_from_weights(vals, rems, need_sum, num_add):
    if num_add == 0:
        v = 0+(need_sum==0)
        return([vals,v])
    if need_sum <= 0:
        res = vals[:]
        res[0] += num_add
        return([res, 0])
    for i in range(len(rems)):
        res = vals[:]
        ind = rems[i][1]
        res[ind] += 1
        [res, v] = update_rems_from_weights(res, rems[i+1:], need_sum-ind, num_add-1)
        if v == 1:
            return([res, v])
    return([vals,0])
    

# age_distrib - list of triplets, each triplet has percentage, min_age, max_age (not inclusive)
# house_size_distrib - list of probs of house size (first is zero), last is probability of that size or more
# percentage_65 - percentage of households with someone over 65
# percentage_child - percentage of households with someone under 18
# n_house = number of households
# tot_pop - total population
# perc_sing_house - percentage of single parent house (out of house with children)
# perc_sing_child - percentage of children that live in single parent home
# child_per_house - distribution of number of children in households with children (if known)

def sim_houses(age_distrib, house_size_distrib, percentage_65, percentage_child, org_n_house, tot_pop, perc_sing_house, perc_sing_child, child_per_house = None, print_fail=0):
    # Get size probabilities that give correct average household size
    [prb_sizes, n_house] = normalize_house_sizes(tot_pop, org_n_house, house_size_distrib)
    
    # Get discrete age probabilities
    rem_ages = get_rem_ages(age_distrib, tot_pop)

    # Get percentage of over 65 households with 2 of them
    percentage_65_szs = get_prob_65(rem_ages, percentage_65, n_house, tot_pop)

    # Get probabilities of single parent households
    sing_sz_probs = get_sing_probs(rem_ages, perc_sing_house, perc_sing_child, tot_pop/(n_house+0.0))
    sing_real = [i*percentage_child*perc_sing_house for i in sing_sz_probs]
    sing_real[0] += 1-sum(sing_real)

    # Get counters of all the stats we have
    rem_ages = from_probs_to_nums(rem_ages, tot_pop)
    rem_65 = from_probs_to_nums(percentage_65_szs, n_house, sum(rem_ages[65:]))
    rem_child = from_probs_to_nums([1-percentage_child, percentage_child], n_house)
    rem_sing = [0]+from_probs_to_nums(sing_real, n_house)[1:]
    rem_sizes = from_probs_to_nums(prb_sizes, n_house, tot_pop)

    if print_fail:
        print ('at start', rem_ages, rem_sizes, rem_sing, rem_child, rem_65)
    # Generate single parent households
    res = []
    # Loop over number of children (0 has 0 probability)
    for k in range(len(rem_sing)):
        v = rem_sing[k]
        for i in range(v):
            # We do iterations, because sometimes we might make a choice
            # that is no longer possible. If this happens too many times
            # we give upm, and this might cause statistical problems. Currently
            # we don't report this, just move on
            niter = 0
            while(1):
                niter += 1
                if niter > 200:
                    if print_fail:
                        print("singles", i, v, k, rem_ages, rem_sizes, rem_child, rem_65, rem_sing)

                    break
                # n_c is number of children, n_a number of adults, n_65 number
                # of people over 65
                n_c = k
                n_a = 1
                n_65 = rand_distrib(rem_65)
                # Is household size still available
                if n_65+n_a+n_c >= len(rem_sizes):
                    continue
                average_adults = sum(rem_ages[65:])*tot_pop/(sum(rem_sizes)*sum(rem_ages)+0.0)
                extra_a = rand_extra_a(n_a+n_65, average_adults, n_c, rem_sizes)
                if rem_sizes[n_65+n_a+n_c+extra_a] == 0:
                    continue
                
                tmp = gen_ages(n_65, n_a, n_c, rem_ages, extra_a)
                if tmp == -1:
                    continue
                break
            if niter <= 200:
                if len(tmp) != n_65+n_a+n_c+extra_a:
                    print ("hmmm", n_65, n_a, extra_a, n_c, tmp)
                res.append(tmp)
                if print_fail > 1:
                    print ('sing', tmp)
                update_rems(tmp, rem_ages, rem_65, rem_child, rem_sing, rem_sizes)
            else:
                break
            
    if print_fail:
        print ('after sing', rem_ages, rem_sizes, rem_sing, rem_child, rem_65)
    # Generate households with children 
    while(rem_child[1]):
        niter = 0
        while(1):
            niter += 1
            if niter > 200:
                if print_fail:
                    print("child", rem_ages, rem_sizes, rem_child, rem_65, rem_sing)
                break
            n_a = 2
            n_65 = rand_distrib(rem_65)
            n_spare = sum([(i-2)*rem_sizes[i] for i in range(3, len(rem_sizes))])-sum(rem_ages[:18])
            # n_spare is number of spare children if we always choose smallest household.
            # If this is zero we cannot afford people over 65
            if n_65 > n_spare:
                if (print_fail):
                    if (niter > 190):
                        print ("n_spare=", n_spare, n_65, rem_child)
                if (niter < 190):
                    continue
            # While we have households over size 6 we need to fill them first
            if sum(rem_sizes[6:]) and niter<155:
                n_c = rand_distrib([0]*6+rem_sizes[6:])
            else:
                n_c = rand_distrib([0]*(n_a+n_65+1)+rem_sizes[n_a+n_65+1:])

            n_c -= (n_a+n_65)
            if n_c <= 0:
                if (print_fail):
                    if (niter > 150):
                        print ("n_c<=0", n_c, n_a, n_65)
                continue
            if (sum(rem_sizes[6:]) == 0):
                # Check if we are taking too many children
                avg_c = sum(rem_ages[:18])/(0.0+rem_child[1])
                if n_c > avg_c*2:
                    if (print_fail):
                        if (niter > 150):
                            print ("n_c>avg", n_c, avg_c)
                    if (niter < 190):
                        continue
            if n_65 + n_a + n_c >= len(rem_sizes):
                if (print_fail):
                    if (niter > 150):
                        print ("big sum", n_c, n_a, n_65, len(rem_sizes))
                continue
            average_adults = sum(rem_ages[65:])*tot_pop/(sum(rem_sizes)*sum(rem_ages)+0.0)
            extra_a = rand_extra_a(n_a+n_65, average_adults, n_c, rem_sizes)
            if rem_sizes[n_65+n_a+n_c+extra_a] == 0:
                if (print_fail):
                    if (niter > 150):
                        print ("empty cell", n_c, n_a, n_65, extra_a, len(rem_sizes))
                continue
                
            # Here we get minimal and maximal number of children we will need.
            # If number of remaining children is problematic, take minimal or
            # maximal size of household remaining
            [mc, mxc, min_ind, max_ind] = get_min_max_child(rem_sizes, rem_child[1])
            rcc = sum(rem_ages[:18])
            if rcc-n_c < rem_child[1]:
                if (print_fail):
                    if (niter > 150):
                        print ("rcc", rcc, n_c, n_a, n_65, extra_a, rem_child)
                if (niter < 190):
                    continue
            #if (mc >= rcc):
            #    if n_c > min_ind-2:
            #        continue
            if (mxc <= rcc):
                if n_c < max_ind-2:
                    if (print_fail):
                        if (niter > 150):
                            print ("mxc", mxc, rcc, n_c, n_a, n_65, extra_a, rem_child, max_ind)
                    if (niter > 190):
                        extra_a = 0
                        n_65 = 0
                        n_c = max_ind-n_a-n_65-extra_a
                    else:
                        continue
                
            tmp = gen_ages(n_65, n_a, n_c, rem_ages, extra_a)
            if tmp == -1:
                if (print_fail):
                        if (niter > 150):
                            print ("bad_gen", mxc, rcc, n_c, n_a, n_65, extra_a, rem_ages)
                continue
            
            break
        
        if niter <= 200:
            if len(tmp) != n_65+n_a+n_c+extra_a:
                    print ("hmmm", n_65, n_a, extra_a, n_c, tmp)
            res.append(tmp)
            if print_fail > 1:
                    print ('child', tmp, rem_ages, rem_sizes, rem_child)
            update_rems(tmp, rem_ages, rem_65, rem_child, rem_sing, rem_sizes)
        else:
            break
    if print_fail:
        print ('after child', rem_ages, rem_sizes, rem_sing, rem_child, rem_65)   
        
    # Generate households with at least 2 over 65 occupants
    for k in range(2, len(rem_65)):
        v = rem_65[k]
        for i in range(v):
            niter = 0
            while(1):
                niter += 1
                if niter > 200:
                    if print_fail:
                        print("pair 65", i, v, k, rem_ages, rem_sizes, rem_child, rem_65, rem_sing)
                    break
                n_65 = k
                n_c = 0
                n_a = rand_distrib([0]*(k)+rem_sizes[k:])
                n_a -= k
                if n_a < 0:
                    n_a = 0
                if n_65+n_a+n_c >= len(rem_sizes):
                    continue
                if rem_sizes[n_65+n_a+n_c] == 0:
                    continue
                tmp = gen_ages(n_65, n_a, n_c, rem_ages)
                if tmp == -1:
                    continue
                break
            if niter <= 200:
                res.append(tmp)
                update_rems(tmp, rem_ages, rem_65, rem_child, rem_sing, rem_sizes)
            else:
                break


    # Generate remaining households
    v = sum(rem_sizes)
    for i in range(v):
        
        niter = 0
        while(1):
            niter += 1
            if niter > 400:
                if print_fail:
                    print("remain", i, v, rem_ages, rem_sizes, rem_child, rem_65, rem_sing)    
                break
            n_65 = rand_distrib(rem_65)
            n_c = 0
            n_a = rand_distrib([0]*n_65+rem_sizes[n_65:])
            n_a -= n_65
            if n_65 + n_a + n_c >= len(rem_sizes):
                continue
            if rem_sizes[n_65+n_a+n_c] == 0:
                continue
            tmp = gen_ages(n_65, n_a, n_c, rem_ages)
            if tmp == -1:
                continue
            break
        if niter <= 400:
            res.append(tmp)
            update_rems(tmp, rem_ages, rem_65, rem_child, rem_sing, rem_sizes)
        else:
            break
        
    if (print_fail):
        print(rem_ages, rem_sizes, rem_child, rem_65, rem_sing)    
    return(res)

# Generate ages given number of each sub population
# if more than one adult or more than one 65+ will assume age difference < 5
# Return -1 if not possible to generate
def gen_ages(n_65, n_a, n_c, rem_ages, extra_a = 0):
    
    if sum(rem_ages[:18]) < n_c:
        return(-1)
    # Get children ages and maximal age
    res = [rand_distrib(rem_ages[:18]) for j in range(n_c)]
    if n_c <= 0:
        mx_c = -1
    else:
        assert len(res) > 0, "%s -- %s " % (len(rem_ages), n_c)
        mx_c = max(res)

    # Get ages of people over 65
    if n_65:
        if sum(rem_ages[65:]) < n_65:
            return(-1)
        niter = 0
        while(1):
            niter += 1
            if niter > 200:
                return(-1)
            v = [rand_distrib(rem_ages[65:])+65 for j in range(n_65)]
            # If there are 2, make sure they are at most 5 years apart
            if max(v)-min(v) > 5:
                continue
            break
        res += v[:]

    # Get ages of adults
    if n_a:
        if sum(rem_ages[18:65]) < n_a:
            return(-1)
        niter = 0
        sum_rem_ages_18_65 = [0]
        for elem in rem_ages[18:65]:
            sum_rem_ages_18_65.append(sum_rem_ages_18_65[-1] + elem)
        while(1):
            niter += 1
            if niter > 400:
                return(-1)
            v = [rand_distrib(rem_ages[18:65], sum_rem_ages_18_65)+18 for j in range(n_a)]
            # If there are more than 1, make sure they are at most 5 years apart
            if max(v)-min(v) > 5:
                continue
            # If there are children, make sure younger parent is
            # between 20 and 40 older that oldest child
            if mx_c != -1:
                d = min(v)-mx_c
                if d<20 or d>40:
                    continue
            break
        res += v
        
    if extra_a:
        all_a = [i for i in res if i >=18]
        if len(all_a) == 0:
            return(-1)
        ma = min(all_a)
        if ma < 38:
            return(-1)
        ma -= 20
        if ma > 25:
            ma = 25
        if sum(rem_ages[18:ma+1]) < extra_a:
            return(-1)
        res += [rand_distrib(rem_ages[18:ma+1])+18 for i in range(extra_a)]
    
    return(res)

# Check how many extra adults we can add
# randomly choose with at most average*2
def rand_extra_a(n_a, average_adults, n_c, rem_sizes):
    if average_adults <= 0:
        return(0)
    max_ext = max([i for i in range(len(rem_sizes)) if rem_sizes[i]])-n_a-n_c
    if max_ext <= 0:
        return(0)
    if max_ext > average_adults*2:
        max_ext = int(average_adults*2)
        
    return(int(random.random()*(max_ext+1)))

# Get probability of each age given data
# Currently we take each group and divide its probility by its size, so all
# ages in the group have same probability
def get_rem_ages(age_distrib, tot_p):
    m = max([i[2] for i in age_distrib])
    res = [0]*m
    for i in age_distrib:
        cur_n = i[2]-i[1]+0.0
        for j in range(i[1], i[2]):
            res[j] += i[0]/cur_n
    res = from_probs_to_nums(res, tot_p)
    return(res)

# Given a household, update counters
def update_rems(agg, rem_ages, rem_65, rem_child, rem_sing, rem_sizes):
    for i in agg:
        rem_ages[i] -= 1
    has_65 = sum([i>= 65 for i in agg])
    rem_65[has_65] -= 1
    nc = sum([i<18 for i in agg])
    has_child = 0+(nc > 0)
    rem_child[has_child] -= 1
    n_a = sum([17<i<65 for i in agg])
    if (n_a == 1) and has_child:
        rem_sing[nc] -= 1
    rem_sizes[len(agg)] -= 1
   

# Given age distribution and percentage of houses with over 65, get
# probability of over 65's 
def get_prob_65(age_distrib, percentage_65, n_house, tot_pop):
    if percentage_65 == 0:
        return([1])
    tot_65 = tot_pop*sum(age_distrib[65:])/(0.0+sum(age_distrib))
    nh_65 = n_house*percentage_65
    if tot_65 <= nh_65:
        return [n_house - tot_65, tot_65 ]
    vals = get_geom_distrib(nh_65, tot_65)
    res = [n_house-nh_65]+vals

# For now assume we have equal probs for first sizes and the last needed
# size finishes the correct sum
#    last_v = 2*tot_65/(max_num+0.0)-nh_65
#    first_v = (nh_65-last_v)/(max_num-1.0)
#    res = [n_house-nh_65]+[first_v]*(max_num-1)+[last_v]
    return(res)

# Get a geometric distribution so that the sum is s and the weighted sum
# (starting with weight first_ind) is ws
# need_dec tells us if series must be decreasing (if fct is bigger than 1 we
# increase length of series)
def get_geom_distrib(s, ws, first_ind = 1, need_dec = 1):
    k = int(ws/s)-first_ind+2
    while(1):
        fct = get_fct(ws/s, k, first_ind = first_ind)
        if fct <= 1 or need_dec==0:
            break
        k +=1 
        if k > 100:
            break
    vals = get_nums_from_fct(fct, s, k)
    return(vals)

# Given the factor between consecutive entries, and the total number
# get the actual numbers
def get_nums_from_fct(f, n, k):
    [s, g, sd, gd] = get_fct_sum(f, k)
    a = n/g
    res = []
    for i in range(k):
        res.append(a)
        a *= f
    
    return(res)
    

# We look for a set of numbers that are a geometric series of
# length k (starting with first_ind), such that their weighted sum divided by
# their sum is x
def get_fct(x, k, first_ind = 1):
    f = 1
    niter = 0
    while(1):
        if niter > 100:
            return(f)
        niter += 1
        [s, g, sd, gd] = get_fct_sum(f, k, first_ind)
        v = s/g-x
        if abs(v) < 0.00001:
            return(f)
        v_dev = sd/g-(s*gd)/(g*g)
        f -= v/v_dev

# Calculate the sum and weighted sum of the geometric series,
# starting the weights from first_ind
def get_fct_sum(f, k, first_ind=1):
    s = 0
    sd = 0
    g = 0
    gd = 0
    v = 1.0
    for i in range(k):
        s += (i+first_ind)*v
        g += v
        v *= f
    v = 1.0/f
    for i in range(k):
        sd += (i+first_ind)*i*v
        gd += i*v
        v *= f
    return([s, g, sd, gd])
# We are given distribution of house sizes, but max size is unlimited. We want
# to produce a distribution that is limited, is the same for low values and
# gives the correct average house size
def normalize_house_sizes(tot_pop, n_house, org_nh_dist):
    s = sum(org_nh_dist)+0.0
    house_size_distrib = [i/s for i in org_nh_dist]
    n_p = sum([i*house_size_distrib[i] for i in range(len(house_size_distrib))])
    q0 = tot_pop/(0.0+n_house)
    if n_p >= q0:
        new_nh = int(tot_pop/(n_p+0.0)+0.9999)
        return([house_size_distrib[:], new_nh])
    max_known = max([i for i in range(len(house_size_distrib)) if house_size_distrib[i]])
    q1 = sum([i*house_size_distrib[i] for i in range(max_known)])
    r_p = house_size_distrib[max_known]
    dq = q0-q1
    vals = get_geom_distrib(r_p, dq, max_known)
    if len(vals) <= 5:
        res = house_size_distrib[:max_known] + vals
        new_nh = n_house
    else:
        res = house_size_distrib[:max_known] + [r_p/5]*5
        n_p = sum([i*res[i] for i in range(len(res))])
        new_nh = int(tot_pop/(n_p+0.0)+0.9999)
    
    return([res, new_nh])

# From stats on single households get prob of 1 or 2 children
def get_sing_probs(rem_ages, perc_sing_house, perc_sing_child, avg_sz):
    pc = sum(rem_ages[:18])/(0.0+sum(rem_ages))
    msc = avg_sz*pc*perc_sing_child/(0.0+perc_sing_house)
    if msc <= 1:
        return([0, 1])
    k = int(2*msc)
    alpha = (k-msc)*2.0/(k*(k-1.0))
    beta = 1-(k-1)*alpha
    return([0]+[alpha]*(k-1)+[beta])

# Given remaining households sizes, and number of households with children
# caculate smallest and largest number of children required, and return also
# smallest and largest relevant sizes that are available
def get_min_max_child(rem_sizes, rem_ch):
    p = 3
    nc0 = 0
    rc = rem_ch
    min_ind = -1
    while(p < len(rem_sizes)):
        cur_h = rem_sizes[p]
        if cur_h and min_ind == -1:
            min_ind = p
        if cur_h > rc:
            cur_h = rc
        nc0 += (p-2)*cur_h
        rc -= cur_h
        p += 1
    nc1 = 0
    p = len(rem_sizes)-1
    rc = rem_ch
    max_ind = -1
    while(p >= 3):
        cur_h = rem_sizes[p]
        if cur_h and max_ind == -1:
            max_ind = p
        if cur_h > rc:
            cur_h = rc
        nc1 += (p-2)*cur_h
        rc -= cur_h
        p -= 1
    return([nc0, nc1, min_ind, max_ind])
    
def test_res_sim(sim_res, age_distrib, house_size_distrib, percentage_65, percentage_child, org_n_house, tot_pop, perc_sing_house, perc_sing_child, child_per_house = None):
    
    # Get size probabilities that give correct average household size
    [prb_sizes, n_house] = normalize_house_sizes(tot_pop, org_n_house, house_size_distrib)

    # Get discrete age probabilities
    rem_ages = get_rem_ages(age_distrib, tot_pop)

    # Get percentage of over 65 households with 2 of them
    percentage_65_szs = get_prob_65(rem_ages, percentage_65, n_house, tot_pop)

    # Get probabilities of single parent households
    sing_sz_probs = get_sing_probs(rem_ages, perc_sing_house, perc_sing_child, tot_pop/(n_house+0.0))
    sing_real = [i*percentage_child*perc_sing_house for i in sing_sz_probs]
    sing_real[0] += 1-sum(sing_real)

    # Get counters of all the stats we have
    rem_ages = from_probs_to_nums(rem_ages, tot_pop)
    rem_65 = from_probs_to_nums(percentage_65_szs, n_house)
    rem_child = from_probs_to_nums([1-percentage_child, percentage_child], n_house)
    rem_sing = [0]+from_probs_to_nums(sing_real, n_house)[1:]
    rem_sizes = from_probs_to_nums(prb_sizes, n_house)
    
    real_ages = [0]*len(rem_ages)
    real_65 = [0]*len(rem_65)
    real_child = [0]*len(rem_child)
    real_sing = [0]*len(rem_sing)
    real_sizes = [0]*len(rem_sizes)
    real_pop = 0
    real_nh = len(sim_res)
    for agg in sim_res:
        real_pop += len(agg)
        for i in agg:
            real_ages[i] += 1
        has_65 = sum([i>= 65 for i in agg])
        real_65[has_65] += 1
        nc = sum([i<18 for i in agg])
        has_child = 0+(nc > 0)
        real_child[has_child] += 1
        n_a = sum([17<i<65 for i in agg])
        if (n_a == 1) and has_child:
            real_sing[nc] += 1
        real_sizes[len(agg)] += 1
        
    print ("ages", rem_ages, [rem_ages[i]-real_ages[i] for i in range(len(rem_ages))])
    print ("sizes", rem_sizes, [rem_sizes[i]-real_sizes[i] for i in range(len(rem_sizes))])
    print ("population", tot_pop, real_pop)
    print ("nh", org_n_house, real_nh)
    
def test_sim(age_distrib, house_size_distrib, percentage_65, percentage_child, n_house, tot_pop, perc_sing_house, perc_sing_child):
    sim_res = sim_houses(age_distrib, house_size_distrib, percentage_65, percentage_child, n_house, tot_pop, perc_sing_house, perc_sing_child)
    test_res_sim(sim_res, age_distrib, house_size_distrib, percentage_65, percentage_child, n_house, tot_pop, perc_sing_house, perc_sing_child)