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
def local_search_v0_iter(A, F, S_in, cost_in):
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
            temp_cost = np.sum(A[:, S_d].min(axis=1))
            if temp_cost < cost:
                cost = temp_cost
                S[i] = v
            #end if
        #end for
    #end for
    return cost, S, iters
#end local_search_v3_iter()

def local_search_v0(A, C, k, seed):
    np.random.seed(seed)

    t, n = C.shape

    F = np.array([], dtype=int)
    for i in range(t):
        F_i = np.array(np.nonzero(C[i])[0])
        F   = np.concatenate([F, F_i])
    #end for
    F = np.sort(F)
    S = np.sort(np.random.choice(F, k))

    cost = np.sum(A[:, S].min(axis=1))
    iters = 0
    while(1):
        cur_cost, cur_S, cur_iters = local_search_v0_iter(A, F, S, cost)
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
    seed_list = [ 94236883, 2611535, 34985942, 6378810, 15208894, 25557092,\
                  43871896, 15786068, 86513484, 118111772]

    for dataset in dataset_list:
        #dataset = "heart-switzerland"
        dist_file = "../dataset_c2/%s-distances-l1.csv"%(dataset)
        color_file = "../dataset_c2/%s-colors.csv"%(dataset)

        A = get_distance_matrix(dist_file)
        C = get_color_matrix(color_file)

        for seed in seed_list:
            total_time = 0.0
            cost = np.inf

            tstart = time.time()
            cur_cost, cur_S, cur_iters =  local_search_v0(A, C, k, seed)
            total_time += time.time() - tstart

            if cur_cost < cost:
                cost = cur_cost
                S = cur_S
            #end if
            
            S = np.sort(S)
            R_d = C[:, S].sum(axis=1)
            output.write("%s, %d, %d, %d, %.2f, %.2f, %d\n"%\
                          (dataset, k, R_d[0], R_d[1], cost,\
                           total_time, seed))
            output.flush()
            sys.stdout.write("%s, %d, %d, %d, %.2f, %.2f, %d\n"%\
                          (dataset, k, R_d[0], R_d[1], cost,\
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
    seed_list = [ 94236883, 2611535, 34985942, 6378810, 15208894, 25557092,\
                  43871896, 15786068, 86513484, 118111772]

    for dataset in dataset_list:
        dist_file = "../dataset_c2/%s-distances-l1.csv"%(dataset)
        color_file = "../dataset_c2/%s-colors.csv"%(dataset)

        A = get_distance_matrix(dist_file)
        C = get_color_matrix(color_file)

        for seed in seed_list:
            total_time = 0.0
            cost = np.inf

            tstart = time.time()
            cur_cost, cur_S, cur_iters =  local_search_v0(A, C, k, seed)
            total_time += time.time() - tstart

            if cur_cost < cost:
                cost = cur_cost
                S = cur_S
            #end if
            
            S = np.sort(S)
            R_d = C[:, S].sum(axis=1)
            output.write("%s, %d, %d, %d, %.2f, %.2f, %d\n"%\
                          (dataset, k, R_d[0], R_d[1], cost,\
                           total_time, seed))
            output.flush()
            sys.stdout.write("%s, %d, %d, %d, %.2f, %.2f, %d\n"%\
                          (dataset, k, R_d[0], R_d[1], cost,\
                           total_time, seed))
            sys.stdout.flush()
            #print(cost, S, total_time)
        #end for
    #end for
#end run_exp1()



################################################################################

def main():
    output = open("results/report-local-search-v0-exp1.out", 'w')
    run_exp1(output)
    output.close()

    output = open("results/report-local-search-v0-exp2.out", 'w')
    run_exp2(output)
    output.close()
#end main()

if __name__=="__main__":
    main()
