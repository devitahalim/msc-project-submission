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
import sys
import time
import matplotlib.pyplot as plt
from functions import read_in_AP, plot_scheduled_routes


# Read data from Excel file
currentDirectory = '/Users/devitahalim/Documents/GitHub/VRPT_Project/prm/'
dataDirectory = '/Users/devitahalim/Documents/GitHub/VRPT_Project/data/'

# file_name = 'Swap_data_5nodes_Instance1'
time_limit = 12 * 3600
bigM=10000

# PARAMETERS
TC = 1 # Transhipment function cost
LCd = 40/60 # Labour cost per minute
VCf = 15 # This can be varied in the result analysis

run = False
load = True
analyse = True
plot = True

# run = True
# load = False
# analyse = False
# plot = False

instanceList = ['toy(chance_constraint)_1B']
# # instanceList = ['AuPost_Instance0B']
# instanceList = ['Aupost_Northern_2019']
instanceList = ['AuPost_Northern_2019','AuPost_Central_2019','AuPost_Western_2019', 'AuPost_Southern_2019'] 
instanceList = ['AuPost_Northern_2019_Mixed','AuPost_Central_2019_Mixed','AuPost_Western_2019_Mixed', 'AuPost_Southern_2019_Mixed'] 


def gurobiOpt(x_bound):

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
    delay = m.addVars(numVehicles, numNodes, n_scenario, vtype = GRB.CONTINUOUS, lb =-GRB.INFINITY, name = 'delay')
    
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
    m.addConstrs((gp.quicksum([y[v, i, j] for j in M]) == gp.quicksum([y[v, j, i] for j in M])
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
        
    
    ### VEHICLE TIMES (SCHEDULED)
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
                 gp.quicksum([gp.quicksum([z[v, n, j, i]*(vArrival_scheduled[v,i] +  expected_loadings[i]) for j in M])
                              for v in V]) for i in M for n in N if i!= productOD[n,0]), name = 'productArrival')
    
    ### Transhipment synchrionisation
    m.addConstrs((prodArrival_scheduled[n, i]<= prodDeparture_scheduled[n,i] for i in M for n in N
                  if i!= productOD[n,0] and i!= productOD[n,1]), name = 'transhipmentSync')
    
    ### UNLOADING requirement
    m.addConstrs((unload[n, i] >= gp.quicksum([z[v, n, k, i] for k in M])
                  - gp.quicksum([z[v, n, i, j] for j in M])
                  for n in N for i in M for v in V if i!= productOD[n,1]), name = 'unloadReq')
    
    ### VEHICLE CAPACITY
    m.addConstrs((gp.quicksum([np.sum(np.fromiter((scenario_dict[w][0]*productSize[scenario_dict[w][2],n] for w in Omega), dtype = float)) * z[v, n, i, j] for n in N]) <= vehicles[v, 1]
                  for i in M for j in M for v in V if i != j), name ='vehicleCap')
    ### DEPOT CAPACITY
    m.addConstrs((gp.quicksum([np.sum(np.fromiter((scenario_dict[w][0]*productSize[scenario_dict[w][2],n] for w in Omega), dtype = float)) * unload [n, i] for n in N]) <= nodeStorage[i]
                  for i in M), name = 'nodeStorage')
    

    ### Actual arrival and departure times
    ### VEHICLE TIMES
    # Departures at non-depot and depot
    m.addConstrs((vDeparture[v, i, w] >= vArrival[v, i, w] +loadings[scenario_dict[w][3],i] for i in M for v in V
                  for w in Omega if i != vehicles[v,0]), name = 'vehicleDeparture')
    
    # Arrivals
    m.addConstrs((vArrival[v, i, w] == gp.quicksum([y[v, j, i] * (vDeparture[v, j, w] + times[scenario_dict[w][1],j,i]) for j in M])
                  for i in M
                  for v in V for w in Omega), name="vArr")
    
    ### PRODUCT TIMES
    # Departures
    m.addConstrs((prodDeparture[n, i, w] ==
                  gp.quicksum([gp.quicksum([z[v, n, i, j] * vDeparture[v,i, w] for j in M])
                               for v in V]) for i in M for n in N  for w in Omega if i!= productOD[n,1]), name = 'productDeparture')
    
    # Arrivals
    m.addConstrs((prodArrival[n, i, w] ==
                 gp.quicksum([gp.quicksum([z[v, n, j, i]*(vArrival[v,i,w] + loadings[scenario_dict[w][3],i]) for j in M])
                              for v in V]) for i in M for n in N for w in Omega if i!= productOD[n,0]), name = 'productArrival')
    
    ### Transhipment synchrionisation
    m.addConstrs((prodArrival[n, i, w]<= prodDeparture[n,i, w] for i in M for n in N  for w in Omega
                  if i!= productOD[n,0] and i!= productOD[n,1]), name = 'transhipmentSync')
    
    ### DELAYED OR NOT
    m.addConstrs((delay[v,i,w]==(vArrival[v,i,w]- vArrival_scheduled[v,i]) for i in M for v in V for w in Omega), name='defineDelay')
    m.addConstrs((vDeparture[v, i, w] >= vDeparture_scheduled[v, i] for i in M for v in V for w in Omega))
    m.addConstrs(vDeparture[v, vehicles[v, 0], w] == vDeparture_scheduled[v, vehicles[v, 0]] for v in V for w in Omega)

    m.modelSense = GRB.MINIMIZE
    m.update()

    #############
    ### OBJECTIVES
    #############

    m.setObjective(TC * gp.quicksum([x[i] for i in M]) +
                   LCd * gp.quicksum((vArrival_scheduled[v, vehicles[v,0]] for v in V))+
                   VCf * gp.quicksum([gp.quicksum([y[v, i, vehicles[v,0]] for i in M]) for v in V]) +
                    (gp.quicksum(scenario_dict[w][0]*
                                 (LCd * gp.quicksum(delay[v, vehicles[v, 0], w] for v in V) + 
                                  VCd[scenario_dict[w][3]] * gp.quicksum(nodeDistances[i, j] * y[v, i, j] for j in M for i in M for v in V)) 
                                  for w in Omega))) # Penalise delayed arrival, not early
    
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

        for i in M:
            x_value[i] = x[i].x
            for v in V:
                arrScheduled_value[v, i] = vArrival_scheduled[v, i].x
                depScheduled_value[v, i] = vDeparture_scheduled[v, i].x
                for j in M:
                    y_value[v, i, j] = y[v, i, j].x
                    for n in N:
                        z_value[v, n, i, j] = z[v, n, i, j].x
                for n in N:
                    unload_value[n, i] = unload[n, i].x

                for w in Omega:
                    arr_value[v, i, w] = vArrival[v, i, w].x
                    dep_value[v, i, w] = vDeparture[v, i, w].x
                    delay_value[v, i, w] = delay[v, i, w].x

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
        objVal = float(0.000)
        # feasibility_array=0
        print('Model is infeasible')

    return [objVal, x_value, y_value, z_value, arrScheduled_value, depScheduled_value, arr_value, dep_value, unload_value, delay_value]  #

for file_name in instanceList:

    mod = 'Stochastic'

    # Read instance file
    scenario_dict, n_scenario, numVehicles, \
        numNodes, vehicles, nPar, nodeDistances, \
            nodeStorage,times, loadings, productOD, productSize, VCd,expected_times, expected_loadings = read_in_AP(dataDirectory, file_name, mod)
    
    case = file_name.split('_')[-1] 
    transhipment_coefs = [[0, 0, 0, 0, 0],[1, 1, 1, 1, 1]]

    if run:
        for tr_case in [0,1]:
            transhipment_coef= transhipment_coefs[tr_case]
            print(f'RUNNING {case}, tr_case {tr_case}')
            time_start = time.time()
            x_array = np.zeros((numNodes))
            y_array = np.zeros(( numVehicles, numNodes, numNodes))
            z_array = np.zeros((numVehicles, numNodes * (numNodes - 1), numNodes, numNodes))
            v_arrScheduled_array = np.zeros ((numVehicles, numNodes))
            v_depScheduled_array = np.zeros((numVehicles, numNodes))
            v_arr_array = np.zeros((numVehicles, numNodes, n_scenario))
            v_dep_array = np.zeros((numVehicles, numNodes, n_scenario))
            unload_array = np.zeros((nPar, numNodes))
            delay_array = np.zeros((numVehicles, numNodes, n_scenario))

            [obj_value, x_array, y_array, z_array, v_arrScheduled_array, v_depScheduled_array, v_arr_array, v_dep_array, unload_array, delay_array] = gurobiOpt(transhipment_coef)

            time_end = time.time()
            print("Takes " + str(time_end - time_start) + " seconds")

            resultFileName= str(tr_case) + "PRM" + file_name  + mod + ".npz"
            resultPath = currentDirectory + 'results/'

            np.savez(os.path.join(resultPath, resultFileName), x=x_array, y=y_array, z=z_array, obj=obj_value, v_a_scheduled = v_arrScheduled_array, v_d_scheduled = v_depScheduled_array,
                        v_a=v_arr_array, v_d=v_dep_array, unload = unload_array, delay = delay_array, cmp=(time_end - time_start))
    

    if analyse:
        for tr_case in [0,1]:
            print(f'Transhipment case {tr_case}, {file_name}')
            if load:
                load_file = currentDirectory + "results/" + str(tr_case) + "PRM" + file_name + mod+ ".npz"
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
                cmp_time = data['cmp']
            # print(delay_array) 
            total_distance = 0
            for s in range(n_scenario):
                for v in range(numVehicles):
                    for i in range(numNodes):
                        for j in range(numNodes):
                            total_distance += scenario_dict[s][0] * y_array[v, i, j]*nodeDistances[i, j]
            print("Total travel distance is: " + str(total_distance))
    
            if plot:
                color_array = ["black", "red", "blue", "darkgreen", "purple", "yellow", "orange", "cyan", "teal", "chocolate",
                                "salmon", "olive", "darkkhaki", "crimson"]
                plt.rcParams.update({'font.size': 22})

                save_dir = currentDirectory +'plots'

                for s in range(n_scenario): #  selected scenarios
                    previous_at = [0, 0]  # (time, node)
                    fig, ax = plt.subplots(figsize=(12,10))
                    plot_scheduled_routes(ax, vehicles, dep_sch_array, arr_sch_array, y_array, color_array, numVehicles, numNodes)
                    for v in range(numVehicles):
                        if sum(v_arr_array[v,:,s]) == 0:
                            continue
                        at = int(vehicles[v, 0])
                        previous_at[1] = at
                        previous_at[0] = v_dep_array[v, at, s]
                        ax.plot([0, previous_at[0]], [previous_at[1] + 1, previous_at[1] + 1], color=color_array[v+1], label='Vehicle'+str(v+1))
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
                    
                    ax.set_title(f'Scenario {s}: Traffic {traffic}, Demand {dem}, Loading {load}')
                    ax.set_title(f'Scenario {s} with index: Demand {scenario_dict[s][2]}, traffic {scenario_dict[s][1]}, loading {scenario_dict[s][3]} ')
                    ax.set_xlabel('Time (min)', size=18)
                    ax.set_ylabel('Node', size=18)
                    ax.legend()
                    ax.grid()

                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    plt.savefig(os.path.join(save_dir, f'{tr_case}_PRM_{file_name}_scenario_{s}_plot.png'))
                    plt.close(fig)

                # plt.show()

