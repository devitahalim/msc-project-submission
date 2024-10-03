"""
Two-stage stochastic program with predetermined route.
Master problem: Determine route for 'normal' scenario and schedule arrival and departure
Subproblem: Determine the actual arrival and departure time
"""

import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import os
import time
import matplotlib.pyplot as plt
from functions import read_in_AP, plot_scheduled_routes


# Read data from Excel file
dataDirectory = '/Users/devitahalim/Documents/GitHub/VRPT_Project/data/'
currentDirectory = '/Users/devitahalim/Documents/GitHub/VRPT_Project/chance_constraint/'

# file_name = 'Swap_data_5nodes_Instance1'
time_limit = 12 * 3600
bigM = 1e4

# PARAMETERS  
TC = 1 # Transhipment function cost
LCd = 40/60 # Labour cost per minute
VCf = 15

run = False
load = True
analyse = True
plot = True

# run = True
# load = False
# analyse = False
# plot = False


# transhipment_coefs = [[0,0,0,0,0], [0,0,0,0,1], [1,1,1,1,1]]


# instanceList = ['AuPost_Instance0B']
# instanceList = ['toy(chance_constraint)_MVeh']
# instanceList = ['AuPost_Western_2019','AuPost_Southern_2019']
# instanceList = ['AuPost_Northern_2019','AuPost_Central_2019'] 
# # instanceList = ['AuPost_Central_2019']
instanceList = ['AuPost_Northern_2019_Mixed','AuPost_Central_2019_Mixed','AuPost_Western_2019_Mixed', 'AuPost_Southern_2019_Mixed'] 
# instanceList = ['AuPost_Northern_2019_C','AuPost_Central_2019_C'] 
alpha_list = [1, 0.99, 0.98, 0.95]
# alpha_list = [1, 0.99, 0.98, 0.95, 0.9, 0.85]
# frequency_list = [3, 5, 10,20]
# alpha_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0, 1.01]

def gurobiOpt(x_bound, alpha):

    global scenario_dict, numVehicles, nPar, n_scenario, numNodes, nodeDistances, nodeStorage, vehicles
    global times, loadings, productSize, productOD, VCd, expected_times, expected_loadings

    # Define sets
    M = range(numNodes)
    V = range(numVehicles)
    N = range(nPar)
    Omega = range(n_scenario)
    ########
    # GUROBI MODEL
    ########
            
    m = gp.Model('PredeterminedRoute_2stage')

    ### VARIABLES
    x = m.addVars(numNodes, vtype = GRB.BINARY, name = 'x') # transhipment node
    y = m.addVars(numVehicles, numNodes, numNodes, vtype = GRB.BINARY, name = 'y') # vehicle route
    z = m.addVars(numVehicles, nPar, numNodes, numNodes, vtype = GRB.BINARY, name = 'z') # parcel route

    vArrival_scheduled = m.addVars(numVehicles, numNodes, vtype = GRB.CONTINUOUS, lb = 0.0, name = 'vArrival_Scheduled')
    vDeparture_scheduled = m.addVars(numVehicles, numNodes, vtype = GRB.CONTINUOUS, lb = 0.0, name = 'vDeparture_Scheduled')
    
    prodArrival_scheduled = m.addVars(nPar, numNodes, vtype = GRB.CONTINUOUS, lb = 0.0, name = 'prodArrival_Scheduled')
    prodDeparture_scheduled = m.addVars(nPar, numNodes, vtype = GRB.CONTINUOUS, lb = 0.0, name = 'prodDeparture_Scheduled')

    vArrival = m.addVars(numVehicles, numNodes, n_scenario, vtype = GRB.CONTINUOUS, lb = 0.0, name = 'vArrival')
    vDeparture = m.addVars(numVehicles, numNodes, n_scenario, vtype = GRB.CONTINUOUS, lb = 0.0, name = 'vDeparture')
    
    prodArrival = m.addVars(nPar, numNodes, n_scenario, vtype = GRB.CONTINUOUS, lb = 0.0, name = 'prodArrival')
    prodDeparture = m.addVars(nPar, numNodes,n_scenario, vtype = GRB.CONTINUOUS, lb = 0.0, name = 'prodDeparture')

    unload = m.addVars(nPar, numNodes, vtype = GRB.BINARY, name = 'unload')
    delay = m.addVars(numVehicles, numNodes, n_scenario, vtype = GRB.CONTINUOUS, lb = -GRB.INFINITY, name = 'delay')
    cap_chance = m.addVars(numVehicles, numNodes, numNodes, n_scenario, vtype = GRB.BINARY, name = 'chance')

    # Actual routing variables
    y_prime = m.addVars(numVehicles, numNodes, numNodes, n_scenario, vtype = GRB.BINARY, name = 'y_prime') # vehicle route
    z_prime = m.addVars(numVehicles, nPar, numNodes, numNodes,n_scenario, vtype = GRB.BINARY, name = 'z_prime') # parcel route

    # d = m.addVars(numNodes, numNodes, n_scenario, vtype = GRB.BINARY, name = 'demand_existance')# d_i,j,omega= 1 if there is demand from i to j
    d = m.addVars(nPar, n_scenario, vtype = GRB.BINARY, name = 'demand_existance')# d_i,j,omega= 1 if there is demand from i to j
    d_prime = m.addVars(numNodes, n_scenario, vtype = GRB.BINARY , name = 'demand_to_node_existance') # d'_i,omega= 1 if there is demand to i
    q = m.addVars(numVehicles, numNodes, vtype = GRB.BINARY, name = 'planned_vehicle_depart_or_not') # q_v,i = 1 if v depart from i (planned)
    q_scenario = m.addVars(numVehicles, numNodes, n_scenario, vtype = GRB.BINARY, name = 'actual_vehicle_depart_or_not')
    demand = m.addVars(nPar, n_scenario, vtype = GRB.CONTINUOUS, lb = 0.0, name = 'demand_in_model') # 0 if CC=0, demand size if CC= 1
    cc = m.addVars(numVehicles, n_scenario, vtype = GRB.BINARY, name = 'chance_per_vehicle')
    cc_prime = m.addVars(nPar, n_scenario, vtype = GRB.BINARY, name = 'chance_per_task')
    cc_2prime = m.addVars(n_scenario, vtype = GRB.BINARY, name = 'chance_per_scenario')

    e = m.addVars(numNodes, numNodes, n_scenario, vtype = GRB.BINARY, name = 'emergency_route')
    ###############
    ### CONSTRAINTS
    ###############

    # Transhipment bound
    m.addConstrs((x[i] <= x_bound[i] for i in M ), name = 'xBound')

    # Eliminate vehicle and product loops on nodes
    m.addConstrs((y[v, i, i] == 0 for v in V for i in M), name = 'vehicleSELF_zero')
    m.addConstrs((z[v, n, i, i] == 0 for v in V for n in N for i in M), name = 'productSELF_zero')

    # Unloading at destination is NOT transhipment
    m.addConstrs((unload[n, productOD[n, 1]] == 0 for n in N), name = 'destinationTrans_ZERO')
    m.addConstrs((unload[n, productOD[n, 0]] == 0 for n in N), name = 'originTrans_ZERO')

    ### VEHICLE FLOW
    # Visit each node at most once
    m.addConstrs((gp.quicksum([y[v, i, j] for j in M]) <= 1 for i in M for v in V), name = 'visitOnce')

    # Flow balance
    m.addConstrs((gp.quicksum([y[v, i, j] for j in M]) == gp.quicksum([y[v, k, i] for k in M])
                  for i in M for v in V), name = 'vehicleFlowBalance')
    
    # Product vehicle connection
    m.addConstrs((z[v, n, i, j] <= y[v, i, j] for v in V for n in N for i in M for j in M
                  if i!=j), name = 'productVehicle')
    
    ### PRODUCT FLOW 
    for n in N:
        # Pick up at origin, Dropped off at destination
        m.addConstr(gp.quicksum([gp.quicksum([z[v, n , productOD[n,0], j] for j in M])
                                for v in V]) == 1, name = 'productOrigin%d' % n)
        m.addConstr(gp.quicksum([gp.quicksum([z[v, n, j, productOD[n,1]] for j in M])
                                for v in V]) == 1 , name = 'productDestination%d' % n)
        
        # Product not travel earlier than origin nor further than destination
        m.addConstr(gp.quicksum([gp.quicksum([z[v, n, j, productOD[n,0]] for j in M])
                                for v in V]) == 0, name = 'productStart%d' % n)
        m.addConstr(gp.quicksum([gp.quicksum([z[v, n, productOD[n,1], j] for j in M])
                                for v in V]) == 0, name ='productEnd%d' % n)
        
        # Product visit at most once at intermediate nodes
        m.addConstrs((gp.quicksum([gp.quicksum([z[v, n, i, j] for j in M]) for v in V]) <=1
                                for i in M if i!= productOD[n,0] 
                                and i!= productOD[n,1]), name ='productVisit%d' %n)
        
        # Flow balance
        m.addConstrs((gp.quicksum([gp.quicksum([z[v, n, i, j] for j in M]) for v in V]) == 
                      gp.quicksum([gp.quicksum([z[v, n, k, i] for k in M]) for v in V])
                    for i in M if i!= productOD[n,0] and i!= productOD[n,1]), name ='productFlowBalance%d' %n)
        
    
    ### VEHICLE TIMES (SCHEDULED) Assuming the expected
    # Departures at non-depot and depot
    m.addConstrs((vDeparture_scheduled[v, i] >= vArrival_scheduled[v, i] +  expected_loadings[i] for i in M for v in V
                   if i != vehicles[v,0]), name = 'vehicleDeparture_scheduled')
    
    # Arrivals
    m.addConstrs((vArrival_scheduled[v, i] == gp.quicksum([y[v, j, i] * (vDeparture_scheduled[v, j] + expected_times[j,i]) for j in M])
                  for i in M
                  for v in V), name="vehicleArrival_scheduled")
    
    # Only tranship on nodes with transhipment function
    m.addConstrs((unload[n, i] <= x[i] for n in N for i in M
                  if i!= productOD[n,0] and i!= productOD[n,1]), name = 'transhipmentONLY_unload')
    
    ### PRODUCT TIMES (SCHEDULED)
    # Departures
    m.addConstrs((prodDeparture_scheduled[n, i] ==
                  gp.quicksum([gp.quicksum([z[v, n, i, j] * vDeparture_scheduled[v,i] for j in M])
                               for v in V]) for i in M for n in N if i!= productOD[n, 1]), name = 'productDeparture_scheduled')
    
    # Arrivals
    m.addConstrs((prodArrival_scheduled[n, i] ==
                 gp.quicksum([gp.quicksum([z[v, n, j, i]*(vArrival_scheduled[v,i] + expected_loadings[i]) for j in M])
                              for v in V]) for i in M for n in N if i!= productOD[n,0]), name = 'productArrival')
    
    ### Transhipment synchrionisation
    m.addConstrs((prodArrival_scheduled[n, i]<= prodDeparture_scheduled[n,i] for i in M for n in N
                  if i!= productOD[n,0] and i!= productOD[n,1]), name = 'transhipmentSync')
    
    ### UNLOADING requirement
    m.addConstrs((unload[n, i] >= gp.quicksum([z[v, n, k, i] for k in M])
                  - gp.quicksum([z[v, n, i, j] for j in M])
                  for n in N for i in M for v in V if i!= productOD[n,1]), name = 'unloadReq')
    

    ### Actual arrival and departure times
    ### VEHICLE TIMES
    # Departures at non-depot and depot
    m.addConstrs((vDeparture[v, i, w] >= vArrival[v, i, w] +loadings[scenario_dict[w][3],i] for i in M for v in V
                  for w in Omega if i != vehicles[v,0]), name = 'vehicleDeparture')
    
    # Arrivals
    m.addConstrs((vArrival[v, i, w] == gp.quicksum([y_prime[v, j, i, w] * (vDeparture[v, j, w] + times[scenario_dict[w][1],j,i]) for j in M])
                  for i in M
                  for v in V for w in Omega), name="vArr")
    
    ### PRODUCT TIMES
    # Departures
    m.addConstrs((prodDeparture[n, i, w] ==
                  gp.quicksum([gp.quicksum([z_prime[v, n, i, j, w] * vDeparture[v,i, w] for j in M])
                               for v in V]) for i in M for n in N  for w in Omega if i!= productOD[n,1]), name = 'productDeparture')
    
    # Arrivals
    m.addConstrs((prodArrival[n, i, w] ==
                 gp.quicksum([gp.quicksum([z_prime[v, n, j, i, w]*(vArrival[v,i,w] + loadings[scenario_dict[w][3],i]) for j in M])
                              for v in V]) for i in M for n in N for w in Omega if i!= productOD[n,0]), name = 'productArrival')
    
    ### Transhipment synchrionisation
    m.addConstrs((prodArrival[n, i, w]<= prodDeparture[n, i, w] for i in M for n in N  for w in Omega
                  if i!= productOD[n,0] and i!= productOD[n,1]), name = 'transhipmentSync')
    
    ### DELAYED OR NOT
    m.addConstrs((delay[v,i,w]==(vArrival[v,i,w]- vArrival_scheduled[v,i])* cc_2prime[w] for i in M for v in V for w in Omega), name='defineDelay')
    m.addConstrs((vDeparture[v, i, w] >= vDeparture_scheduled[v, i] for i in M for v in V for w in Omega))

    ### DEPOT CAPACITY
    m.addConstrs((gp.quicksum([productSize[scenario_dict[w][2],n] * unload [n, i] for n in N]) <= nodeStorage[i]
                  for i in M for w in Omega), name = 'nodeStorage')
    
    # Depart from depot as scheduled (so there is no delayed departure)
    m.addConstrs(vDeparture[v, vehicles[v, 0], w] == vDeparture_scheduled[v, vehicles[v, 0]] for v in V for w in Omega)

    ### ACTUAL ROUTING 
    
    # # Define d_n,w = 1 if task n is done, 0 if not
    m.addConstrs(demand[n, w] <= bigM* d[n, w] for n in N for w in Omega)
    m.addConstrs(demand[n, w] >= d[n, w] for n in N for w in Omega)

    # Define demand
    m.addConstrs(demand[n, w] == productSize[scenario_dict[w][2], n] * cc_2prime[w] for n in N for w in Omega)
    
    # routing (no y_prime)
    m.addConstrs(y_prime[v, i, j, w] == y[v, i, j] * cc_2prime[w] for i in M for j in M for v in V for w in Omega)
    m.addConstrs(z_prime[v, n, i, j, w] <= y_prime[v, i, j, w] for i in M for j in M for v in V for n in N for w in Omega)
    m.addConstrs(z_prime[v, n, i, j, w] == z[v, n, i, j]* d[n, w] for i in M for j in M for v in V for n in N for w in Omega)

    # CHANCE CONSTRAINT
    m.addConstrs((gp.quicksum(productSize[scenario_dict[w][2],n]* z[v,n,i,j] for n in N)- vehicles[v,1]) <= bigM * (1 - cap_chance[v, i, j, w])
                  for i in M for j in M for v in V for w in Omega)
    
    m.addConstrs((vehicles[v,1] - gp.quicksum(productSize[scenario_dict[w][2],n]* z[v,n,i,j]for n in N)) <= bigM * cap_chance[v, i, j, w]
                 for i in M for j in M for v in V for w in Omega)
    
    m.addConstrs(cap_chance[v, i, i, w] == 1 for v in V for i in M for w in Omega)
    m.addConstrs(cc_2prime[w] <= cap_chance[v, i, j, w] for v in V for i in M for j in M for w in Omega)
    m.addConstrs(cc_2prime[w] >= gp.quicksum(cap_chance[v, i, j, w] for v in V for i in M for j in M)-(numNodes*numNodes * numVehicles-1) for w in Omega)
    m.addConstr((gp.quicksum(scenario_dict[w][0]* cc_2prime[w] for w in Omega)  >= alpha))


    m.modelSense = GRB.MINIMIZE
    m.update()

    #############
    ### OBJECTIVES
    #############
    
    # No emergency cost
    m.setObjective(TC * gp.quicksum([x[i] for i in M])+ LCd * gp.quicksum((vArrival_scheduled[v, vehicles[v,0]] for v in V)) # including waiting time at the start
                   + gp.quicksum(scenario_dict[w][0]*
                                 (VCd[scenario_dict[w][3]] * gp.quicksum(nodeDistances[i, j] * y_prime[v, i, j, w] for j in M for i in M for v in V)+
                                  VCf * gp.quicksum(y_prime[v, i, vehicles[v,0], w] for i in M for v in V)+ 
                                  LCd * gp.quicksum(delay[v, vehicles[v, 0], w] for v in V))
                                   for w in Omega)) # Penalise delayed arrival, not early
    
    
    #############
    ### OPTIMISE
    #############

    m.setParam('TimeLimit', time_limit)
    m.optimize()

    if m.status == GRB.OPTIMAL:
        x_value = np.zeros(numNodes)
        y_value = np.zeros((numVehicles, numNodes, numNodes))
        z_value = np.zeros((numVehicles, nPar, numNodes, numNodes))

        arrScheduled_value = np.zeros((numVehicles, numNodes))
        depScheduled_value = np.zeros((numVehicles, numNodes))

        arr_value = np.zeros((numVehicles, numNodes, n_scenario))
        dep_value = np.zeros((numVehicles, numNodes, n_scenario))
        unload_value = np.zeros((nPar, numNodes))

        delay_value = np.zeros((numVehicles, numNodes, n_scenario))
        cap_value = np.zeros((numVehicles, numNodes, numNodes, n_scenario))

        y_prime_value = np.zeros((numVehicles, numNodes, numNodes, n_scenario))
        z_prime_value = np.zeros((numVehicles, nPar, numNodes, numNodes, n_scenario))
        d_value = np.zeros((nPar, n_scenario))

        # d_value = np.zeros((numNodes, numNodes, n_scenario))
        d_prime_value = np.zeros((numNodes, n_scenario))
        q_value = np.zeros((numVehicles, numNodes))
        q_scenario_value = np.zeros((numVehicles, numNodes, n_scenario))
        demand_value = np.zeros((nPar, n_scenario))
        cc_value = np.zeros((numVehicles, n_scenario))
        cc_prime_value = np.zeros((nPar, n_scenario))
        cc_2prime_value = np.zeros((n_scenario))
        e_value =  np.zeros((numNodes, numNodes, n_scenario))

        for i in M:
            x_value[i] = x[i].x

            for n in N:
                unload_value[n, i] = unload[n, i].x

            for w in Omega:
                d_prime_value[i ,w] = d_prime[i, w].x

                for j in M:
                    e_value[i, j, w] = e[i, j, w].x

            
            for v in V:
                arrScheduled_value[v, i] = vArrival_scheduled[v, i].x
                depScheduled_value[v, i] = vDeparture_scheduled[v, i].x
                q_value[v, i] = q[v, i].x
                for j in M:
                    y_value[v, i, j] = y[v, i, j].x

                    for n in N:
                        z_value[v, n, i, j] = z[v, n, i, j].x
                        for w in Omega:
                            z_prime_value[v, n, i, j, w] = z_prime[v, n, i, j, w].x

                    for w in Omega:
                        cap_value[v, i, j, w] = cap_chance[v, i, j, w].x
                        y_prime_value[v, i, j, w] = y_prime[v, i, j, w].x
            

                for w in Omega:
                    arr_value[v, i, w] = vArrival[v, i, w].x
                    dep_value[v, i, w] = vDeparture[v, i, w].x
                    delay_value[v, i, w] = delay[v, i, w].x
                    q_scenario_value[v, i, w] = q_scenario[v, i, w].x

        for w in Omega:
            cc_2prime_value[w] = cc_2prime[w].x
            for v in V:
                cc_value[v, w] = cc[v, w].x
            for n in N:
                cc_prime_value[n, w] = cc_prime[n, w].x
                demand_value[n, w] = demand[n, w].x
                d_value [n, w] = d[n, w].x
                
                

        obj = m.getObjective()
        objVal= obj.getValue()
    
    else:
        x_value = np.zeros(numNodes)
        y_value = np.zeros((numVehicles, numNodes, numNodes))
        z_value = np.zeros((numVehicles, nPar, numNodes, numNodes))
        arrScheduled_value = np.zeros((numVehicles, numNodes))
        depScheduled_value = np.zeros((numVehicles, numNodes))
        arr_value = np.zeros((numVehicles, numNodes, n_scenario))
        dep_value = np.zeros((numVehicles, numNodes, n_scenario))
        unload_value = np.zeros((nPar, numNodes))
        delay_value = np.zeros((numVehicles, numNodes, n_scenario))
        cap_value = np.zeros((numVehicles, numNodes, numNodes, n_scenario))

        y_prime_value = np.zeros((numVehicles, numNodes, numNodes, n_scenario))
        z_prime_value = np.zeros((numVehicles, nPar, numNodes, numNodes, n_scenario))
        # d_value = np.zeros((numNodes, numNodes, n_scenario))
        d_value = np.zeros((nPar, n_scenario))
        d_prime_value = np.zeros((numNodes, n_scenario))
        q_value = np.zeros((numVehicles, numNodes))
        q_scenario_value = np.zeros((numVehicles, numNodes, n_scenario))
        demand_value = np.zeros((nPar, n_scenario))
        cc_value = np.zeros((numVehicles, n_scenario))
        cc_prime_value = np.zeros((nPar, n_scenario))
        cc_2prime_value = np.zeros((n_scenario))
        e_value = np.zeros((numNodes, numNodes, n_scenario))

        objVal = float(0.000)
        # feasibility_array=0
        print('Model is infeasible')

    return [objVal, x_value, y_value, z_value, arrScheduled_value, depScheduled_value, arr_value, dep_value, unload_value, delay_value, cap_value,\
            y_prime_value, z_prime_value, d_value, d_prime_value, q_value, q_scenario_value, demand_value, cc_value, cc_prime_value, cc_2prime_value, e_value]  #

for file_name in instanceList:
    mod = 'Stochastic'
    # Read instance file
    scenario_dict, n_scenario, numVehicles, numNodes, vehicles, nPar, nodeDistances, nodeStorage,times, loadings, productOD, productSize, VCd,expected_times, expected_loadings= read_in_AP(dataDirectory, file_name, mod)
    # scenario_dict, n_scenario, numVehicles, numNodes, vehicles, nPar, nodeDistances, nodeStorage,times, loadings, productOD, productSize = read_in(dataDirectory, file_name)
    
    transhipment_coefs = [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]

    case = file_name.split('_')[-2]
    if file_name.split('_')[-1]=='Mixed' or  file_name.split('_')[-1]=='C':
        case = file_name.split('_')[-3]

    if case == 'Northern':
        transhipment_coefs = [[0, 0, 0], [1, 0, 0], [1, 1, 1]]
    elif case == 'Central':
        transhipment_coefs = [[0, 0, 0], [1, 0, 0], [1, 1, 1]]
    elif case == 'Western':
        transhipment_coefs = [[0, 0, 0, 0], [0, 1, 0, 1], [1,1,1,1]]
    elif case == 'Southern':
        transhipment_coefs = [[0, 0, 0, 0], [1, 0, 0, 0], [1,1,1,1]]
    
    if run:
        for alpha in alpha_list:
            for tr_case in [0, 2]:
                print('---------------------------------------')
                print(f'Running tr_case {tr_case}, with alpha {alpha}:')
                print('---------------------------------------')
                transhipment_coef= transhipment_coefs[tr_case]
                time_start = time.time()
                x_array = np.zeros((numNodes))
                y_array = np.zeros((numVehicles, numNodes, numNodes))
                z_array = np.zeros((numVehicles, numNodes * (numNodes - 1), numNodes, numNodes))
                v_arrScheduled_array = np.zeros ((numVehicles, numNodes))
                v_depScheduled_array = np.zeros((numVehicles, numNodes))
                v_arr_array = np.zeros((numVehicles, numNodes, n_scenario))
                v_dep_array = np.zeros((numVehicles, numNodes, n_scenario))
                unload_array = np.zeros((nPar, numNodes))
                delay_array = np.zeros((numVehicles, numNodes, n_scenario))
                cap_array = np.zeros((numVehicles, numNodes, numNodes, n_scenario))
                

                y_prime_array = np.zeros((numVehicles, numNodes, numNodes, n_scenario))
                z_prime_array = np.zeros((numVehicles, nPar, numNodes, numNodes, n_scenario))
                # d_array = np.zeros((numNodes, numNodes, n_scenario))
                d_array = np.zeros((nPar, n_scenario))
                d_prime_array = np.zeros((numNodes, n_scenario))
                q_array = np.zeros((numVehicles, numNodes))
                q_scenario_array = np.zeros((numVehicles, numNodes, n_scenario))
                demand_array = np.zeros((nPar, n_scenario))
                cc_array = np.zeros((numVehicles, n_scenario))
                cc_prime_array = np.zeros((nPar, n_scenario))
                cc_2_prime_array = np.zeros((n_scenario))
                e_array = np.zeros((numNodes, numNodes, n_scenario))

                [obj_value, x_array, y_array, z_array, v_arrScheduled_array, v_depScheduled_array, \
                v_arr_array, v_dep_array, unload_array, delay_array,cap_array, y_prime_array, z_prime_array, d_array,
                d_prime_array, q_array, q_scenario_array, demand_array, cc_array, cc_prime_array, cc_2_prime_array, e_array\
                    ] = gurobiOpt(transhipment_coef, alpha)

                time_end = time.time()
                print("Takes " + str(time_end - time_start) + " seconds")

                resultFileName= str(tr_case) +'NEC_1'+'_alpha'+str(alpha)+ file_name  + ".npz"
                resultPath = currentDirectory + 'results_omit_failure/'

                np.savez(os.path.join(resultPath, resultFileName), x=x_array, y=y_array, z=z_array, obj=obj_value, v_a_scheduled = v_arrScheduled_array, v_d_scheduled = v_depScheduled_array,
                            v_a=v_arr_array, v_d=v_dep_array, unload = unload_array, delay = delay_array, capacitychance = cap_array, 
                            y_prime = y_prime_array, z_prime = z_prime_array, d = d_array, d_prime = d_prime_array, q = q_array, q_scenario = q_scenario_array,
                            demand_var = demand_array, cc = cc_array, cc_prime = cc_prime_array, cc_2prime = cc_2_prime_array, e = e_array,
                            cmp=(time_end - time_start), alpha = alpha)  

    if analyse:
        for alpha in alpha_list:
            for tr_case in [0,2]:
                print(f'Transhipment case {tr_case}, {file_name}')
                if load:
                    load_file = currentDirectory +'results_omit_failure/' + str(tr_case) +'NEC_1'+'_alpha'+str(alpha)+ file_name  + ".npz"
                    data = np.load(load_file)
                    obj_values = data['obj']
                    x_array = data['x']
                    y_array = data['y']
                    z_array = data['z']
                    arr_sch_array = data['v_a_scheduled']
                    dep_sch_array = data['v_d_scheduled']
                    v_arr_array = data['v_a']
                    v_dep_array = data['v_d']
                    unload_array = data['unload']
                    delay_array = data['delay']
                    cap_array = data['capacitychance']
                    y_prime_array = data['y_prime']
                    z_prime_array = data['z_prime']
                    d_array = data['d']
                    d_prime_array = data['d_prime']
                    q_array = data['q']
                    q_scenario_array = data['q_scenario']
                    demand_array = data['demand_var']
                    cc_array =data['cc']
                    cc_prime_array = data['cc_prime']
                    cc_2_prime_array = data['cc_2prime']
                    e_array = data['e']
                    cmp_time = data['cmp']
                    alpha = data['alpha']
                # total_distance = 0
                # for s in range(n_scenario):
                #     for v in range(numVehicles):
                #         for i in range(numNodes):
                #             for j in range(numNodes):
                #                 total_distance += scenario_dict[s][0] * y_prime_array[v, i, j,s]*nodeDistances[i, j]
                # print("Total travel distance is: " + str(total_distance))
        
                if plot:
                    color_array = ["black", "red", "blue", "darkgreen", "purple", "yellow", "orange", "cyan", "teal", "chocolate",
                                    "salmon", "olive", "darkkhaki", "crimson"]
                    plt.rcParams.update({'font.size': 20})

                    save_dir = currentDirectory + 'plots_omit_failure/'

                    for s in range(n_scenario): #  selected scenarios
                        previous_at = [0, 0]  # (time, node)
                        fig, ax = plt.subplots(figsize=(15, 10))
                        
                        plot_scheduled_routes(ax, vehicles, dep_sch_array, arr_sch_array, y_array, color_array, numVehicles, numNodes)
                        for v in range(numVehicles):
                            if sum(v_arr_array[v,:,s]) == 0:
                                continue
                            at = int(vehicles[v, 0])
                            previous_at[1] = at
                            previous_at[0] = v_dep_array[v, at, s]
                            ax.plot([0, previous_at[0]], [previous_at[1] + 1, previous_at[1] + 1], color=color_array[v+1], label='Vehicle '+str(v+1))
                            # print("Vehicle", v + 1, "start fr", at + 1, "dep", "{:.2f}".format(dep_array[v, at]))
                            while 1:
                                for j in range(numNodes):
                                    if round(y_array[ v, at, j]):
                                        at = j
                                        break
                                if at == vehicles[v, 0]:
                                    now_at = [v_arr_array[v, at, s], at]
                                    ax.plot([previous_at[0], now_at[0]], [previous_at[1] + 1, now_at[1] + 1],
                                            color=color_array[v+1])
                                    break
                                else:
                                    now_at = [v_arr_array[v, at, s], at]
                                    ax.plot([previous_at[0], now_at[0]], [previous_at[1] + 1, now_at[1] + 1], color=color_array[v+1])
                                    ax.plot([v_arr_array[v, at, s], v_dep_array[v, at, s]], [now_at[1] + 1, now_at[1] + 1], color=color_array[v+1])
                                    previous_at[0] = v_dep_array[v, at, s]
                                    previous_at[1] = now_at[1]
                            # ax.('Vehicle ' + str(v + 1))
                                    
                        if scenario_dict[s][2] == 0:
                            dem = 'Low'
                        elif scenario_dict[s][2] == 1:
                            dem = 'Average'
                        else:
                            dem = 'High'

                        if scenario_dict[s][3] == 0:
                            load = 'Slow'
                        elif scenario_dict[s][3] == 1:
                            load = 'Fast'

                        if scenario_dict[s][1] == 0:
                            traffic = 'Slow'
                        elif scenario_dict[s][1] == 1:
                            traffic = 'Average'
                        else:
                            traffic = 'Fast'
                        
                        tolerance = round((1-alpha), 2)
                        if int(cc_2_prime_array[s]) == 0:
                                ax.set_title(f'Scenario {s} (FAILURE): Traffic {traffic}, Demand {dem}, Loading {load}, \u03B1={alpha}')
                        else:
                            ax.set_title(f'Scenario {s}: Traffic {traffic}, Demand {dem}, Loading {load}, \u03B1={alpha}')

                        ax.set_xlabel('Time (min)', size=16)
                        ax.set_ylabel('Node', size=16)
                        # Add a legend
                        pos = ax.get_position()
                        ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
                        ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.7))
                        ax.grid()


                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        plt.savefig(os.path.join(save_dir, f'{tr_case}_NEC_1_{file_name}_alpha_{alpha}_scenario_{s}.png'))
                        plt.close(fig)

                # plt.show()


