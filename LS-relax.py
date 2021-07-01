import numpy as np
import itertools
import sys
import argparse
import os
import random
import tqdm
import time


###############################################################################
def get_distance_matrix(dist_matrix_file):
    tstart = time.time()
    if not os.path.exists(dist_matrix_file):
        sys.stderr.write("File '%s' do not exist\n"%(dist_matrix_file))
    #end if
    dist_matrix = np.loadtxt(fname=dist_matrix_file, delimiter=",", dtype=float)

    #sys.stdout.write("get-distance-matrix: [total: %.2fs]\n"%(time.time()-tstart))
    #sys.stdout.flush()
    return dist_matrix
#end get_distance_matrix()

def get_color_matrix(color_matrix_file):
    tstart= time.time()
    if not os.path.exists(color_matrix_file):
        sys.stderr.write("File '%s' do not exist\n"%(dist_matrix_file))
    #end if
    color_matrix = np.loadtxt(fname=color_matrix_file, delimiter=",", dtype=int)

    #sys.stdout.write("get-color-matrix: [total: %.2fs]\n"%(time.time()-tstart))
    #sys.stdout.flush()
    return color_matrix
#end get_distance_matrix()

################################################################################
def local_search_v3_iter(A, C, R, F, S_in, cost_in):
    cost = cost_in
    S = np.sort(S_in)

    iters = 0
    for i in range(len(S)):
        u = S[i]
        S_d = S.copy()
        for j in range(len(F)):
            v = F[j]
            if v in S_d:
                continue;
            iters += 1

            S_d[i] = v
            R_d = C[:, S_d].sum(axis=1)
            temp_R = np.subtract(R, R_d)

            if np.any(temp_R < 0):
                continue;
            temp_cost = np.sum(A[:, S_d].min(axis=1))
            if temp_cost < cost:
                cost = temp_cost
                S[i] = v
            #end if
        #end for
    #end for

    return cost, S, iters
#end local_search_v3_iter()

def local_search_v3(A, C, R, seed):
    np.random.seed(seed)
    r0 = R[0]
    r1 = R[1]

    F0 = np.array(np.nonzero(C[0])[0])
    F1 = np.array(np.nonzero(C[1])[0])
    F  = np.sort(np.concatenate([F0, F1]))

    if (len(F0) < r0) or (len(F1) < r1):
        cost = np.inf
        solution = []
        return cost, solution, 0
    #end if

    # initialise a random assignment
    S0 = np.random.choice(F0, r0)
    S1 = np.random.choice(F1, r1)
    
    S    = np.sort(np.concatenate([S0, S1]))
    cost = np.sum(A[:, S].min(axis=1))

    iters = 0
    while(1):
        cur_cost, cur_S, cur_iters = local_search_v3_iter(A, C, R, F, S, cost)
        cur_R = C[:, cur_S].sum(axis=1)
        iters += cur_iters

        if cur_cost >= cost:
            break;
        else:
            cost = cur_cost
            S   = cur_S
        #end if
    #end while

    return cost, S, iters
#end local_search_v3()

###############################################################################
def run_exp1(output = sys.stdout):
    dataset_list = ["heart-switzerland",\
                    "heart-va",\
                    "heart-hungarian",\
                    "heart-cleveland",\
                    "student-mat",\
                    "house-votes-84",\
                    "student-por",\
                    "student-per2",\
                    "autism",\
                    "hcv-egy-data",\
                    #"cmc"
                    ]
    k = 10
    min_frac_list = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    #min_frac_list = np.array([0.1, 0.2])
    #seed_list = [ 94236883, 2611535, 34985942, 6378810, 15208894, 25557092,\
    #              43871896, 15786068, 86513484, 118111772]
    seed_list = [123456789]

    for dataset in dataset_list:
        #dataset = "heart-switzerland"
        dist_file = "../dataset_c2/%s-distances-l1.csv"%(dataset)
        color_file = "../dataset_c2/%s-colors.csv"%(dataset)

        A = get_distance_matrix(dist_file)
        C = get_color_matrix(color_file)

        for seed in seed_list:
            for min_frac in min_frac_list:
                total_time = 0.0
                cost = np.inf
                r0_min = int(k*min_frac)
                for r0 in range(r0_min, k+1):
                    r1 = int(k - r0)
                    R = np.array([r0, r1])

                    tstart = time.time()
                    cur_cost, cur_S, cur_iters =  local_search_v3(A, C, R, seed)
                    total_time += time.time() - tstart

                    if cur_cost < cost:
                        cost = cur_cost
                        S = cur_S
                    #end if
                #end for
                S = np.sort(S)
                R_d = C[:, S].sum(axis=1)
                output.write("%s, %d, %.2f, %d, %d, %.2f, %.2f, %d\n"%\
                              (dataset, k, min_frac, R_d[0], R_d[1], cost,\
                               total_time, seed))
                output.flush()
                sys.stdout.write("%s, %d, %.2f, %d, %d, %.2f, %.2f, %d\n"%\
                              (dataset, k, min_frac, R_d[0], R_d[1], cost,\
                               total_time, seed))
                sys.stdout.flush()
                #print(cost, S, total_time)
        #end for
    #end for
#end run_exp1()


def run_exp2(output = sys.stdout):
    dataset_list = ["cmc",\
                    "abalone",\
                    "mushroom",\
                    "nursery",\
                    "census-income"\
                    ]
    k = 10
    min_frac_list = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    #seed_list = [ 94236883, 2611535, 34985942, 6378810, 15208894, 25557092,\
    #              43871896, 15786068, 86513484, 118111772]
    seed_list = [123456789]

    for dataset in dataset_list:
        #dataset = "heart-switzerland"
        dist_file = "../dataset_c2/%s-distances-l1.csv"%(dataset)
        color_file = "../dataset_c2/%s-colors.csv"%(dataset)

        A = get_distance_matrix(dist_file)
        C = get_color_matrix(color_file)

        for seed in seed_list:
            for min_frac in min_frac_list:
                total_time = 0.0
                cost = np.inf
                r0_min = int(k*min_frac)
                for r0 in range(r0_min, k+1):
                    r1 = int(k - r0)
                    R = np.array([r0, r1])

                    tstart = time.time()
                    cur_cost, cur_S0, cur_S1, cur_iters =  local_search_v3(A, C, R, seed)
                    total_time += time.time() - tstart

                    if cur_cost < cost:
                        cost = cur_cost
                        S0 = cur_S0
                        S1 = cur_S1
                    #end if
                #end for
                S = np.sort(np.concatenate([S0, S1]))
                output.write("%s, %d, %.2f, %d, %d, %.2f, %.2f, %d\n"%\
                              (dataset, k, min_frac, len(S0), len(S1), cost,\
                               total_time, seed))
                output.flush()
                sys.stdout.write("%s, %d, %.2f, %d, %d, %.2f, %.2f, %d\n"%\
                              (dataset, k, min_frac, len(S0), len(S1), cost,\
                               total_time, seed))
                sys.stdout.flush()
                #print(cost, S0, S1, S, total_time)
        #end for
    #end for
#end run_exp1()

################################################################################
def local_search_v3_gen(A, C, R, seed):
    np.random.seed(seed)

    F = np.array([], dtype=int)
    S = np.array([], dtype=int)
    for i in range(len(R)):
        F_i = np.array(np.nonzero(C[i])[0])

        if len(F_i) < R[i]:
            return np.inf, [], 0

        S_i = np.random.choice(F_i, R[i])
        F = np.concatenate([F, F_i])
        S = np.concatenate([S, S_i])
    #end for
    F = np.sort(F)
    S = np.sort(S)

    cost = np.sum(A[:, S].min(axis=1))
    iters = 0
    while(1):
        cur_cost, cur_S, cur_iters = local_search_v3_iter(A, C, R, F, S, cost)
        cur_R = C[:, cur_S].sum(axis=1)
        iters += cur_iters

        if cur_cost >= cost:
            break;
        else:
            cost = cur_cost
            S   = cur_S
        #end if
    #end while

    return cost, S, iters
#end local_search_v3()

def run_exp3(output = sys.stdout):
    dataset_list = ["heart-switzerland",\
                    "heart-va",\
                    "heart-hungarian",\
                    "heart-cleveland",\
                    "student-mat",\
                    "house-votes-84",\
                    "student-por",\
                    "student-per2",\
                    "autism",\
                    "hcv-egy-data",\
                    #"cmc"
                    ]
    k = 10
    min_frac_list = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    seed_list = [ 94236883, 2611535, 34985942, 6378810, 15208894, 25557092,\
                  43871896, 15786068, 86513484, 118111772]

    for dataset in dataset_list:
        #dataset = "heart-switzerland"
        dist_file = "../dataset_c2/%s-distances-l1.csv"%(dataset)
        color_file = "../dataset_c2/%s-colors.csv"%(dataset)

        A = get_distance_matrix(dist_file)
        C = get_color_matrix(color_file)

        for seed in seed_list:
            for min_frac in min_frac_list:
                total_time = 0.0
                cost = np.inf
                r0_min = int(k*min_frac)
                for r0 in range(r0_min, k+1):
                    r1 = int(k - r0)
                    R = np.array([r0, r1])

                    tstart = time.time()
                    cur_cost, cur_S, cur_iters =  local_search_v3_gen(A, C, R, seed)
                    total_time += time.time() - tstart

                    if cur_cost < cost:
                        cost = cur_cost
                        S = cur_S
                    #end if
                #end for
                S = np.sort(S)
                R_d = C[:, S].sum(axis=1)
                output.write("%s, %d, %.2f, %d, %d, %.2f, %.2f, %d\n"%\
                              (dataset, k, min_frac, R_d[0], R_d[1], cost,\
                               total_time, seed))
                output.flush()
                sys.stdout.write("%s, %d, %.2f, %d, %d, %.2f, %.2f, %d\n"%\
                              (dataset, k, min_frac, R_d[0], R_d[1], cost,\
                               total_time, seed))
                sys.stdout.flush()
                #print(cost, S, total_time)
        #end for
    #end for
#end run_exp1()



################################################################################

def main():
    output = open("results/report-local-search-v3-exp3.out", 'w')
    run_exp3(output)
    output.close()

    output = open("results/report-local-search-v1-exp2.out", 'w')
    run_exp2(output)
    output.close()
#end main()

if __name__=="__main__":
    main()
