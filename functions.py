import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import os
import math
import time
import matplotlib.pyplot as plt

def split_task(vehicles, productSize, productOD):
    n_scenario = len(productSize)
    # Initialize lists to hold updated product size and OD arrays
    max_capacity = np.max(vehicles[:, 1])
    # print(max_capacity)

    # New structures to store updated information
    new_sizes = [[] for _ in range(productSize.shape[0])]
    new_ods = []

    # Determine if splitting is needed and process sizes and ODs accordingly
    for col in range(productSize.shape[1]):
        # This will check if any demand in this column requires splitting
        needs_split = any(demand > max_capacity for demand in productSize[:, col])

        column_max_splits = 0  # Track maximum number of splits needed for any row in this column

        # Determine the maximum number of splits required in this column
        for demand in productSize[:, col]:
            splits = int(np.ceil(demand / max_capacity))
            if splits > column_max_splits:
                column_max_splits = splits
                
        
        # Process each row for the current column
        for row in range(productSize.shape[0]):

            demand = productSize[row, col]
            if demand > max_capacity:
                # Calculate splits and assign them
                splits = int(np.ceil(demand / max_capacity))
                split_size = demand / splits
                new_sizes[row].extend([split_size] * splits)
                if splits < column_max_splits:
                    new_sizes[row]. extend([0] * (column_max_splits- splits))
            else:
                # Append the original demand and potentially a zero if there's a split in this column
                new_sizes[row].append(demand)
                if needs_split:
                    new_sizes[row].extend([0] * (column_max_splits-1))

        # Append the correct number of OD entries for this column based on the maximum number of splits
        for _ in range(column_max_splits):
            new_ods.append(productOD[col % productOD.shape[0]])  # Use modulo for cycling through OD if necessary
        
    # Flatten new_sizes to match the expected format and ensure correct column alignment
    updated_productSize = np.array([item for sublist in new_sizes for item in sublist]).reshape(n_scenario, -1)

    # Adjust new_ods if necessary to flatten it correctly
    updated_productOD = np.array(new_ods).reshape(-1, 2)

    return updated_productOD, updated_productSize

def read_location(dataDirectory, file_name):

    df = pd.read_excel(dataDirectory+ file_name + '.xlsx', sheet_name='Stores', usecols=['Longitude', 'Latitude', 'Storage capacity'])
    nodeLocations = df.values[:, 0:2]
    nodeStorage = df.values[:, 2]
    numNodes = len(nodeStorage)

    return nodeLocations, nodeStorage, numNodes

def read_distances(dataDirectory, file_name):

    df = pd.read_excel(dataDirectory+ file_name + '.xlsx', sheet_name='Distances(KM)')
    nodeDistances = df.values

    return nodeDistances

def read_vehicles(dataDirectory, file_name):

    df = pd.read_excel(dataDirectory+ file_name + '.xlsx', sheet_name='Vehicles', index_col=0)
    vehicles = np.transpose(df.values)  # depot, capacity
    numVehicles = len(vehicles)
    for v in range(numVehicles):
        vehicles[v, 0] = vehicles[v, 0] - 1
    
    return vehicles, numVehicles

def read_demand(dataDirectory, file_name, mod, numNodes):
    df = pd.read_excel(dataDirectory + file_name + '.xlsx', sheet_name='Demand(prob)'+mod)
    prob_demand = df.values[0]
    n_sc_demand = len(prob_demand)

    df = pd.read_excel(dataDirectory + file_name + '.xlsx', sheet_name='Demand(items)'+mod)
    assert(np.size(df.values, 0) == numNodes * n_sc_demand)
    demand = np.rint(np.reshape(df.values, (n_sc_demand, numNodes, numNodes)))
    n_parF = numNodes*numNodes - numNodes
    p_odF = np.zeros((n_parF, 2), dtype=int)  # [origin, destination]
    p_sizeF = np.zeros((n_sc_demand, n_parF))
    counter = 0
    for j in range(numNodes):
        for k in range(numNodes):
            if j == k:
                continue
            p_odF[counter, 0] = j
            p_odF[counter, 1] = k
            for i in range(n_sc_demand):
                p_sizeF[i, counter] = demand[i, j, k]
            counter += 1
    expected_size = np.rint(np.mean(p_sizeF, axis=0))

    return prob_demand, n_sc_demand, demand, n_parF, p_odF, p_sizeF, expected_size

def read_speed(dataDirectory, file_name, numNodes, nodeDistances):
    df = pd.read_excel(dataDirectory+file_name + '.xlsx', sheet_name='Speeds(KpH)', index_col=0)
    n_sc_speed = np.size(df.values, 1)
    prob_speed = df.values[1, :]
    speeds = df.values[0, :]
    costs_km = df.values[2, :]
    # Get travel time matrices
    times = np.zeros((n_sc_speed, numNodes, numNodes))
    for i in range(n_sc_speed):
        times[i, :, :] = nodeDistances[:, :]/speeds[i] * 60  # in minutes
    times = np.rint(times)
    expected_times = np.rint(np.mean(times, axis=0))

    return n_sc_speed, prob_speed, speeds, costs_km, times, expected_times
    
def read_loading(dataDirectory, file_name, numNodes):

    df = pd.read_excel(dataDirectory+file_name + '.xlsx', sheet_name='Loading(min)', index_col=0)
    n_sc_loading = np.size(df.values, 0)
    prob_loading = df.values[:, 0]
    loadings = np.rint(df.values[:, 1:numNodes + 1])
    expected_loadings = np.rint(np.mean(loadings, axis=0))

    return n_sc_loading, prob_loading, loadings, expected_loadings


def read_in_AP(dataDirectory, file_name, modeltype, frequency=1):
    if modeltype == 'Stochastic':
        mod =''
    elif modeltype == 'Deterministic':
        mod='_det'
    else:
        return ('Error model type')

    # Read locations and storage capacities of nodes (locations to be fixed later)
    nodeLocations, nodeStorage, numNodes = read_location(dataDirectory, file_name)

    # Read distances
    nodeDistances = read_distances(dataDirectory, file_name)

    # Read vehicles
    vehicles, numVehicles = read_vehicles(dataDirectory, file_name)

    # Read demands and associated probabilities
    prob_demand, n_sc_demand, demand, n_parF, p_odF, p_sizeF, expected_size = read_demand(dataDirectory, file_name, mod, numNodes)

    # Remove the zero demand from the delivery list
    temp_nPar = int(sum(expected_size != 0))
    temp_productOD = np.zeros((temp_nPar, 2), dtype=int)
    temp_productSize = np.zeros((n_sc_demand, temp_nPar))
    
    counter = 0
    for i in range(len(expected_size)):
        if expected_size[i] != 0:
            temp_productOD[counter] = p_odF[i]
            for s in range(n_sc_demand):
                temp_productSize[s, counter] = p_sizeF[s, i]
            counter += 1
    
    temp2_productSize = []
    for s in range(len(temp_productSize)):
        temp2_productSize.append([math.ceil(temp_productSize[s][n]/frequency) for n in range(temp_nPar)])

    temp2_productSize = np.array(temp2_productSize)
    productOD, productSize = split_task(vehicles, temp2_productSize, temp_productOD)
    nPar = len(productOD)

    # Read speeds
    n_sc_speed, prob_speed, speeds, costs_km, times, expected_times = read_speed(dataDirectory, file_name, numNodes, nodeDistances)

    # Read loading times
    n_sc_loading, prob_loading, loadings, expected_loadings = read_loading(dataDirectory, file_name, numNodes)

    # Set scenario dictionary: [prob, time, demand, loading]
    scenario_dict = {}
    counter = 0
    for i in range(n_sc_speed):
        for j in range(n_sc_demand):
            for k in range(n_sc_loading):
                scenario_dict[counter] = [prob_speed[i]*prob_demand[j]*prob_loading[k], i, j, k]
                counter += 1
    n_scenario = len(scenario_dict)

    return (scenario_dict, n_scenario, numVehicles, numNodes, 
            vehicles, nPar, nodeDistances, nodeStorage, 
            times, loadings, productOD, productSize, costs_km, expected_times, expected_loadings)

def plot_scheduled_routes(ax, vehicles, dep_sch_array, arr_sch_array, y_array, color_array, numVehicles, numNodes):
    for v in range(numVehicles):
        if sum(arr_sch_array[v,:]) == 0:
            continue
        at = int(vehicles[v, 0])
        previous_at = [0, at]
        previous_at[0] = dep_sch_array[v, at]
        ax.plot([0, previous_at[0]], [previous_at[1] + 1, previous_at[1] + 1], color=color_array[v+1], linestyle=':', alpha=0.5, label='Scheduled vehicle '+str(v+1))
        while 1:
            for j in range(numNodes):
                if round(y_array[ v, at, j]):
                    at = j
                    break
            if at == vehicles[v, 0]:
                now_at = [arr_sch_array[v, at], at]
                ax.plot([previous_at[0], now_at[0]], [previous_at[1] + 1, now_at[1] + 1], color=color_array[v+1], linestyle=':', alpha=0.5)
                break
            else:
                now_at = [arr_sch_array[v, at], at]
                ax.plot([previous_at[0], now_at[0]], [previous_at[1] + 1, now_at[1] + 1], color=color_array[v+1], linestyle=':', alpha=0.5)
                ax.plot([arr_sch_array[v, at], dep_sch_array[v, at]], [now_at[1] + 1, now_at[1] + 1], color=color_array[v+1], linestyle=':', alpha=0.5)
                previous_at[0] = dep_sch_array[v, at]
                previous_at[1] = now_at[1]




def read_in(dataDirectory, file_name):

    # Read locations and storage capacities of nodes (locations to be fixed later)
    df = pd.read_excel(dataDirectory+ file_name + '.xlsx', sheet_name='Stores', usecols=['Easting', 'Northing', 'Storage capacity'])
    nodeLocations = df.values[:, 0:2]
    nodeStorage = df.values[:, 2]
    numNodes = len(nodeStorage)

    # Read distances
    df = pd.read_excel(dataDirectory+ file_name + '.xlsx', sheet_name='Distances(KM)')
    nodeDistances = df.values

    # Read vehicles
    df = pd.read_excel(dataDirectory+ file_name + '.xlsx', sheet_name='Vehicles', index_col=0)
    vehicles = np.transpose(df.values)  # depot, capacity
    numVehicles = len(vehicles)
    for v in range(numVehicles):
        vehicles[v, 0] = vehicles[v, 0] - 1

    # # Read local route windows
    # df = pd.read_excel(data_dir+file_name, sheet_name='Local windows(hr)', index_col=0)
    # depot_windows = np.transpose(df.values)*60  # arrival, departure of local routes in min

    # Read demands and associated probabilities
    df = pd.read_excel(dataDirectory + file_name + '.xlsx', sheet_name='Demand(prob)')
    prob_demand = df.values[0]
    n_sc_demand = len(prob_demand)

    df = pd.read_excel(dataDirectory + file_name + '.xlsx', sheet_name='Demand(items)')
    assert(np.size(df.values, 0) == numNodes * n_sc_demand)
    demand = np.rint(np.reshape(df.values, (n_sc_demand, numNodes, numNodes)))
    n_parF = numNodes*numNodes - numNodes
    p_odF = np.zeros((n_parF, 2), dtype=int)  # [origin, destination]
    p_sizeF = np.zeros((n_sc_demand, n_parF))
    counter = 0
    for j in range(numNodes):
        for k in range(numNodes):
            if j == k:
                continue
            p_odF[counter, 0] = j
            p_odF[counter, 1] = k
            for i in range(n_sc_demand):
                p_sizeF[i, counter] = demand[i, j, k]
            counter += 1
    expected_size = np.rint(np.mean(p_sizeF, axis=0))

    # Remove the zero demand from the delivery list
    nPar = int(sum(expected_size != 0))
    productOD = np.zeros((nPar, 2), dtype=int)
    productSize = np.zeros((n_sc_demand, nPar))
    counter = 0
    for i in range(len(expected_size)):
        if expected_size[i] != 0:
            productOD[counter] = p_odF[i]
            for s in range(n_sc_demand):
                productSize[s, counter] = p_sizeF[s, i]
            counter += 1

    # Read speeds
    df = pd.read_excel(dataDirectory+file_name + '.xlsx', sheet_name='Speeds(KpH)', index_col=0)
    n_sc_speed = np.size(df.values, 1)
    prob_speed = df.values[1, :]
    speeds = df.values[0, :]
    # Get travel time matrices
    times = np.zeros((n_sc_speed, numNodes, numNodes))
    for i in range(n_sc_speed):
        times[i, :, :] = nodeDistances[:, :]/speeds[i] * 60  # in minutes
    times = np.rint(times)
    expected_times = np.rint(np.mean(times, axis=0))

    # Read loading times
    df = pd.read_excel(dataDirectory+file_name + '.xlsx', sheet_name='Loading(min)', index_col=0)
    n_sc_loading = np.size(df.values, 0)
    prob_loading = df.values[:, 0]
    loadings = np.rint(df.values[:, 1:numNodes + 1])
    expected_loadings = np.rint(np.mean(loadings, axis=0))

    # Set scenario dictionary: [prob, time, demand, loading]
    scenario_dict = {}
    counter = 0
    for i in range(n_sc_speed):
        for j in range(n_sc_demand):
            for k in range(n_sc_loading):
                scenario_dict[counter] = [prob_speed[i]*prob_demand[j]*prob_loading[k], i, j, k]
                counter += 1
    n_scenario = len(scenario_dict)

    return (scenario_dict, n_scenario, numVehicles, numNodes, 
            vehicles, nPar, nodeDistances, nodeStorage, 
            times, loadings, productOD, productSize)
    

def read_in2(dataDirectory, file_name, demand_percentages = [0.75, 1, 2]):
    """
    Read excel data and extract information on vehicles, nodes, task, demands, scenarios.
    """
    # Node locations and capacities 
    df = pd.read_excel(dataDirectory+file_name, sheet_name='Location', usecols=['Longitude', 'Latitude', 'Capacity'])
    nodeLocations = df.values[:, 0:2]
    nodeStorage = df.values[:, 2]
    numNodes = len(nodeStorage)

    # Distances
    df = pd.read_excel(dataDirectory+file_name, sheet_name='Distance(km)', usecols=lambda x: x != 'Unnamed: 0')
    nodeDistances = df.values

    # Vehicles (Depot, capacity)
    df = pd.read_excel(dataDirectory+file_name, sheet_name = 'Vehicles', index_col=0)
    vehicles = np.transpose(df.values)
    numVehicles = len(vehicles)
    for v in range(numVehicles):
        vehicles[v,0] = vehicles[v,0]-1


    # Demands and the probabilities
    df = pd.read_excel(dataDirectory+file_name, sheet_name='Demand(prob)')
    prob_demand = df.values[0]
    n_sc_demand = len(prob_demand)

    df = pd.read_excel(dataDirectory+file_name, sheet_name='Demand', usecols=lambda x: x != 'Unnamed: 0')

    compiled_demand=[]
    for r in demand_percentages:
        demand = df * r
        compiled_demand.append(demand)

    # Concatenate the realizations into one DF
    compiled_df = pd.concat(compiled_demand, ignore_index=True)
    assert(np.size(compiled_df.values, 0) == numNodes * n_sc_demand)
    demand = np.rint(np.reshape(compiled_df.values, (n_sc_demand, numNodes, numNodes)))
    nPar_F = numNodes * numNodes - numNodes
    productOD_F = np.zeros((nPar_F, 2), dtype=int)  # [origin, destination]
    productSize_F = np.zeros((n_sc_demand, nPar_F)) # task size

    counter = 0
    for i in range(numNodes):
        for j in range(numNodes):
            if i == j:
                continue
            productOD_F[counter, 0] = i
            productOD_F[counter, 1] = j
            for r in range(n_sc_demand):
                productSize_F[r, counter] = demand[r, i, j]
            counter += 1

    expected_size = np.rint(np.mean(productSize_F, axis=0))

    # Remove zero demands
    nPar = int(sum(expected_size != 0))
    productOD = np.zeros((nPar, 2), dtype=int)
    productSize = np.zeros((n_sc_demand, nPar))

    counter = 0
    for i in range(len(expected_size)):
        if expected_size[i] != 0:
            productOD[counter] = productOD_F[i]
            for s in range(n_sc_demand):
                productSize[s, counter] = productSize_F[s, i]
            counter += 1

    # Speeds
    df = pd.read_excel(dataDirectory+file_name, sheet_name='Speeds(KpH)', index_col=0)
    n_sc_speed = np.size(df.values, 1)
    prob_speed = df.values[1, :]
    speeds = df.values[0, :]

    # Travel time
    times = times = np.zeros((n_sc_demand, numNodes, numNodes))
    for i in range(n_sc_speed):
        times[i,:,:] = nodeDistances[:,:]/ speeds[i] *60

    times= np.rint(times)
    expected_times = np.rint(np.mean(times, axis = 0))

    # Loading times
    df = pd.read_excel(dataDirectory+file_name, sheet_name='Loading(min)', index_col=0)
    n_sc_loading = np.size(df.values, 0)
    prob_loading = df.values[:, 0]
    loadings = np.rint(df.values[:, 1:numNodes + 1])
    expected_loadings = np.rint(np.mean(loadings, axis=0))

    # Scenario dictionary
    scenario_dict = {}
    counter = 0
    for i in range(n_sc_speed):
        for j in range(n_sc_demand):
            for k in range (n_sc_loading):
                scenario_dict[counter] = [prob_speed[i] * prob_demand[j] * prob_loading[k], i, j, k]
                counter +=1

    n_scenario = len(scenario_dict)

    return (scenario_dict, n_scenario, numVehicles, numNodes, 
            vehicles, nPar, nodeDistances, nodeStorage, 
            times, loadings, productOD, productSize)