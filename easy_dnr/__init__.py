import pandapower as pp
import pandapower.networks as nw
import pandapower.plotting as plot
import pandapower.topology as top

import copy
import os
import sys
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from troma import DitString
from troma import MatchingPursuitResults

def _create_9bus_dnr_network():
    """
    9-bus distribution network for DNR, derived from case33bw parameters.

    Topology (bus indices):
        0(src)-1-2-3-4    main feeder  (case33bw lines 0-3)
               |   |
               5   8      lateral stubs (case33bw lines 17, 21)
               |
               6
               |
               7

    Normally-open tie switches: (4,7) and (7,8).

    n_buses=9, n_switches=10, n_closed=8, n_dits=2
    """
    VN_KV = 12.66
    net = pp.create_empty_network(sn_mva=1.0, f_hz=60.0)

    b = [pp.create_bus(net, vn_kv=VN_KV) for _ in range(9)]
    pp.create_ext_grid(net, bus=b[0], vm_pu=1.0, va_degree=0.0)

    def _line(f, t, r, x):
        pp.create_line_from_parameters(net, from_bus=b[f], to_bus=b[t],
            length_km=1.0, r_ohm_per_km=r, x_ohm_per_km=x,
            c_nf_per_km=0.0, max_i_ka=99999.0, in_service=True)

    def _tie(f, t, r, x):
        pp.create_line_from_parameters(net, from_bus=b[f], to_bus=b[t],
            length_km=1.0, r_ohm_per_km=r, x_ohm_per_km=x,
            c_nf_per_km=0.0, max_i_ka=99999.0, in_service=False)

    # Main feeder (case33bw lines 0-3: buses 0→1→2→3→4)
    _line(0, 1, 0.0922, 0.0470)
    _line(1, 2, 0.4930, 0.2511)
    _line(2, 3, 0.3660, 0.1864)
    _line(3, 4, 0.3811, 0.1941)
    # Lateral 1 from bus 1 (case33bw lines 17-19: buses 1→18→19→20, remapped to 5→6→7)
    _line(1, 5, 0.1640, 0.1565)
    _line(5, 6, 1.5042, 1.3554)
    _line(6, 7, 0.4095, 0.4784)
    # Lateral 2 stub from bus 2 (case33bw line 21: bus 2→22, remapped to 2→8)
    _line(2, 8, 0.4512, 0.3083)

    # Tie switches (normally open)
    _tie(4, 7, 2.0000, 2.0000)  # inspired by case33bw tie 32 (bus 20→7)
    _tie(7, 8, 2.0000, 2.0000)  # inspired by case33bw tie 34 (bus 11→21)

    # Loads (case33bw values for the corresponding original buses)
    load_data = [
        (1, 0.100, 0.060), (2, 0.090, 0.040), (3, 0.120, 0.080),
        (4, 0.060, 0.030), (5, 0.090, 0.040), (6, 0.090, 0.040),
        (7, 0.090, 0.040), (8, 0.090, 0.050),
    ]
    for bus, p, q in load_data:
        pp.create_load(net, bus=b[bus], p_mw=p, q_mvar=q)

    return net


def _create_12bus_dnr_network():
    """
    12-bus distribution network for DNR, derived from case33bw parameters.

    Topology (bus indices):
        Main feeder:           0(src)-1-2-3-4
        Lateral A (from bus 1): 1-5-6-7
        Lateral B (from bus 2): 2-8-9
        Lateral C (from bus 3): 3-10-11

    Normally-open tie switches: (7,9), (9,11), (4,7).

    n_buses=12, n_switches=14, n_closed=11, n_dits=3

    Benchmark rationale: 364 feasible states (C(14,3)) — between the 9-bus (45)
    and 15-bus (680) — with an estimated circuit depth ~200 at p=1, making it
    the most suitable size for QPU experiments on current hardware.
    """
    VN_KV = 12.66
    net = pp.create_empty_network(sn_mva=1.0, f_hz=60.0)

    b = [pp.create_bus(net, vn_kv=VN_KV) for _ in range(12)]
    pp.create_ext_grid(net, bus=b[0], vm_pu=1.0, va_degree=0.0)

    def _line(f, t, r, x):
        pp.create_line_from_parameters(net, from_bus=b[f], to_bus=b[t],
            length_km=1.0, r_ohm_per_km=r, x_ohm_per_km=x,
            c_nf_per_km=0.0, max_i_ka=99999.0, in_service=True)

    def _tie(f, t, r, x):
        pp.create_line_from_parameters(net, from_bus=b[f], to_bus=b[t],
            length_km=1.0, r_ohm_per_km=r, x_ohm_per_km=x,
            c_nf_per_km=0.0, max_i_ka=99999.0, in_service=False)

    # Main feeder (case33bw lines 0-3: buses 0→1→2→3→4)
    _line(0, 1,  0.0922, 0.0470)
    _line(1, 2,  0.4930, 0.2511)
    _line(2, 3,  0.3660, 0.1864)
    _line(3, 4,  0.3811, 0.1941)
    # Lateral A from bus 1 (case33bw lines 17-19: remapped to 1→5→6→7)
    _line(1, 5,  0.1640, 0.1565)
    _line(5, 6,  1.5042, 1.3554)
    _line(6, 7,  0.4095, 0.4784)
    # Lateral B from bus 2 (case33bw lines 21-22: remapped to 2→8→9)
    _line(2, 8,  0.4512, 0.3083)
    _line(8, 9,  0.8980, 0.7091)
    # Lateral C from bus 3 (case33bw lines 24-25: remapped to 3→10→11)
    _line(3,  10, 0.2030, 0.1034)
    _line(10, 11, 0.8960, 0.7011)

    # Tie switches (normally open)
    _tie(7,  9,  2.0000, 2.0000)  # Lateral A tip ↔ Lateral B tip
    _tie(9,  11, 2.0000, 2.0000)  # Lateral B tip ↔ Lateral C tip
    _tie(4,  7,  0.5000, 0.5000)  # Main feeder tip ↔ Lateral A tip

    # Loads (case33bw values for the corresponding original buses)
    load_data = [
        (1,  0.100, 0.060), (2,  0.090, 0.040), (3,  0.120, 0.080),
        (4,  0.060, 0.030), (5,  0.090, 0.040), (6,  0.090, 0.040),
        (7,  0.090, 0.040), (8,  0.090, 0.050), (9,  0.090, 0.050),
        (10, 0.060, 0.025), (11, 0.060, 0.025),
    ]
    for bus, p, q in load_data:
        pp.create_load(net, bus=b[bus], p_mw=p, q_mvar=q)

    return net


def _create_15bus_dnr_network():
    """
    15-bus distribution network for DNR, derived from case33bw parameters.

    Topology (bus indices):
        0(src)-1-2-3-4-5    main feeder       (case33bw lines 0-4)
               |   |   |
               6  10  13    lateral roots     (case33bw lines 17, 21, 24)
               |   |   |
               7  11  14    (laterals continue)
               |   |
               8  12
               |
               9

    Normally-open tie switches: (9,12), (12,14), (5,9).

    n_buses=15, n_switches=17, n_closed=14, n_dits=3
    """
    VN_KV = 12.66
    net = pp.create_empty_network(sn_mva=1.0, f_hz=60.0)

    b = [pp.create_bus(net, vn_kv=VN_KV) for _ in range(15)]
    pp.create_ext_grid(net, bus=b[0], vm_pu=1.0, va_degree=0.0)

    def _line(f, t, r, x):
        pp.create_line_from_parameters(net, from_bus=b[f], to_bus=b[t],
            length_km=1.0, r_ohm_per_km=r, x_ohm_per_km=x,
            c_nf_per_km=0.0, max_i_ka=99999.0, in_service=True)

    def _tie(f, t, r, x):
        pp.create_line_from_parameters(net, from_bus=b[f], to_bus=b[t],
            length_km=1.0, r_ohm_per_km=r, x_ohm_per_km=x,
            c_nf_per_km=0.0, max_i_ka=99999.0, in_service=False)

    # Main feeder (case33bw lines 0-4: buses 0→1→2→3→4→5)
    _line(0, 1,  0.0922, 0.0470)
    _line(1, 2,  0.4930, 0.2511)
    _line(2, 3,  0.3660, 0.1864)
    _line(3, 4,  0.3811, 0.1941)
    _line(4, 5,  0.8190, 0.7070)
    # Lateral 1 from bus 1 (case33bw lines 17-20: 1→18→19→20→21, remapped to 1→6→7→8→9)
    _line(1, 6,  0.1640, 0.1565)
    _line(6, 7,  1.5042, 1.3554)
    _line(7, 8,  0.4095, 0.4784)
    _line(8, 9,  0.7089, 0.9373)
    # Lateral 2 from bus 2 (case33bw lines 21-23: 2→22→23→24, remapped to 2→10→11→12)
    _line(2,  10, 0.4512, 0.3083)
    _line(10, 11, 0.8980, 0.7091)
    _line(11, 12, 0.8960, 0.7011)
    # Lateral 3 stub from bus 5 (case33bw lines 24-25: 5→25→26, remapped to 5→13→14)
    _line(5,  13, 0.2030, 0.1034)
    _line(13, 14, 0.2842, 0.1447)

    # Tie switches (normally open)
    _tie(9,  12, 2.0000, 2.0000)  # inspired by case33bw tie 32 (bus 20→7)
    _tie(12, 14, 2.0000, 2.0000)  # inspired by case33bw tie 33 (bus 8→14)
    _tie(5,   9, 0.5000, 0.5000)  # inspired by case33bw tie 35 (bus 17→32)

    # Loads (case33bw values for the corresponding original buses)
    load_data = [
        (1,  0.100, 0.060), (2,  0.090, 0.040), (3,  0.120, 0.080),
        (4,  0.060, 0.030), (5,  0.060, 0.020), (6,  0.090, 0.040),
        (7,  0.090, 0.040), (8,  0.090, 0.040), (9,  0.090, 0.040),
        (10, 0.090, 0.050), (11, 0.420, 0.200), (12, 0.420, 0.200),
        (13, 0.060, 0.025), (14, 0.060, 0.025),
    ]
    for bus, p, q in load_data:
        pp.create_load(net, bus=b[bus], p_mw=p, q_mvar=q)

    return net


def _create_26bus_dnr_network():
    """
    26-bus distribution network for DNR, derived from case33bw parameters.

    Topology (bus indices):
        Main feeder:            0(src)-1-2-3-4-5-6-7
        Lateral A (from bus 1): 1-8-9-10-11
        Lateral B (from bus 2): 2-12-13-14
        Lateral C (from bus 4): 4-15-16-17
        Lateral D (from bus 6): 6-18-19-20
        Lateral E (from bus 7): 7-21-22-23
        Stub      (from bus 3): 3-24-25

    Normally-open tie switches (5):
        (11,14), (17,20), (23,25), (14,24), (11,23)

    n_buses=26, n_switches=30, n_closed=25, n_dits=5
    """
    VN_KV = 12.66
    net = pp.create_empty_network(sn_mva=1.0, f_hz=60.0)

    b = [pp.create_bus(net, vn_kv=VN_KV) for _ in range(26)]
    pp.create_ext_grid(net, bus=b[0], vm_pu=1.0, va_degree=0.0)

    def _line(f, t, r, x):
        pp.create_line_from_parameters(net, from_bus=b[f], to_bus=b[t],
            length_km=1.0, r_ohm_per_km=r, x_ohm_per_km=x,
            c_nf_per_km=0.0, max_i_ka=99999.0, in_service=True)

    def _tie(f, t, r, x):
        pp.create_line_from_parameters(net, from_bus=b[f], to_bus=b[t],
            length_km=1.0, r_ohm_per_km=r, x_ohm_per_km=x,
            c_nf_per_km=0.0, max_i_ka=99999.0, in_service=False)

    # Main feeder (buses 0→1→2→3→4→5→6→7)
    _line(0, 1,  0.0922, 0.0470)
    _line(1, 2,  0.4930, 0.2511)
    _line(2, 3,  0.3660, 0.1864)
    _line(3, 4,  0.3811, 0.1941)
    _line(4, 5,  0.8190, 0.7070)
    _line(5, 6,  0.1872, 0.6188)
    _line(6, 7,  0.7114, 0.2351)
    # Lateral A from bus 1 (1→8→9→10→11)
    _line(1,  8,  0.1640, 0.1565)
    _line(8,  9,  1.5042, 1.3554)
    _line(9,  10, 0.4095, 0.4784)
    _line(10, 11, 0.7089, 0.9373)
    # Lateral B from bus 2 (2→12→13→14)
    _line(2,  12, 0.4512, 0.3083)
    _line(12, 13, 0.8980, 0.7091)
    _line(13, 14, 0.8960, 0.7011)
    # Lateral C from bus 4 (4→15→16→17)
    _line(4,  15, 0.2030, 0.1034)
    _line(15, 16, 0.2842, 0.1447)
    _line(16, 17, 0.5075, 0.2585)
    # Lateral D from bus 6 (6→18→19→20)
    _line(6,  18, 0.9744, 0.9630)
    _line(18, 19, 0.3100, 0.3600)
    _line(19, 20, 0.5416, 0.7129)
    # Lateral E from bus 7 (7→21→22→23)
    _line(7,  21, 0.7070, 0.9373)
    _line(21, 22, 0.3100, 0.3600)
    _line(22, 23, 0.5416, 0.7129)
    # Stub from bus 3 (3→24→25)
    _line(3,  24, 0.2030, 0.1034)
    _line(24, 25, 0.2842, 0.1447)

    # Tie switches (normally open) — 5 ties
    _tie(11, 14, 2.0000, 2.0000)  # A tip ↔ B tip
    _tie(17, 20, 2.0000, 2.0000)  # C tip ↔ D tip
    _tie(23, 25, 2.0000, 2.0000)  # E tip ↔ stub tip
    _tie(14, 24, 2.0000, 2.0000)  # B tip ↔ stub root
    _tie(11, 23, 2.0000, 2.0000)  # A tip ↔ E tip

    # Loads (case33bw-inspired values)
    load_data = [
        (1,  0.100, 0.060), (2,  0.090, 0.040), (3,  0.120, 0.080),
        (4,  0.060, 0.030), (5,  0.060, 0.020), (6,  0.060, 0.020),
        (7,  0.200, 0.100), (8,  0.200, 0.100), (9,  0.060, 0.020),
        (10, 0.060, 0.020), (11, 0.090, 0.050), (12, 0.045, 0.030),
        (13, 0.060, 0.035), (14, 0.060, 0.035), (15, 0.060, 0.025),
        (16, 0.060, 0.025), (17, 0.120, 0.080), (18, 0.060, 0.025),
        (19, 0.060, 0.025), (20, 0.120, 0.070), (21, 0.200, 0.100),
        (22, 0.060, 0.025), (23, 0.060, 0.025), (24, 0.420, 0.200),
        (25, 0.420, 0.200),
    ]
    for bus, p, q in load_data:
        pp.create_load(net, bus=b[bus], p_mw=p, q_mvar=q)

    return net


def load_network(name):
    """
    Loads the specified network and replaces all lines with switches in the open state.
     - This allows the DNR problem to be represented as a binary configuration of switch states.
     - The original line data is preserved in the pandapower net, but all lines are effectively "removed" from the network topology by opening the corresponding switches.
     - The function currently supports the "33_bus" network, but can be extended to include other predefined pandapower networks or custom network files as needed.
     - The resulting net will have the same buses and lines as the original, but all lines will be in service with an open switch, allowing the DNR problem to be solved by configuring the switch states.

    Parameters
    ----------
    name : str
        The name of the network to load. Supported values include "33_bus",
        "9_bus" (9-bus, 2 tie switches), "12_bus" (12-bus, 3 tie switches,
        14 switches total — recommended QPU benchmark), "15_bus" (15-bus,
        3 tie switches), "26_bus" (26-bus, 5 tie switches, 30 switches total).

    Returns
    -------
    pandapowerNet
        A pandapower network object with all lines replaced by open switches, ready for DNR configuration.
    """

    def _replace_close_by_switch(net):
        for i, line in net.line.iterrows():
            pp.create_switch(
                net,
                bus=line.from_bus,
                element=i,
                et='l',
                closed=True
            )
        for i, line in net.line.iterrows():
            if not line["in_service"]:
                net.switch.at[i, "closed"] = False
                net.line.at[i, "in_service"] = True

    if name == "33_bus":
        net = nw.case33bw()
    elif name == "9_bus":
        net = _create_9bus_dnr_network()
    elif name == "12_bus":
        net = _create_12bus_dnr_network()
    elif name == "15_bus":
        net = _create_15bus_dnr_network()
    elif name == "26_bus":
        net = _create_26bus_dnr_network()
    else:
        raise ValueError(f"Unknown network name: {name}")

    _replace_close_by_switch(net)
    net.switch["closed"] = [False]*len(net.switch)
    return net

def check_radiality(net):
    """
    Radiality check: A network is radial if it is a tree, meaning it has no cycles and is fully connected.
    """
    mg = top.create_nxgraph(net, respect_switches=True)
    g = nx.Graph(mg)

    is_radial = nx.is_tree(g)
    return is_radial

def check_connectivity(net):
    """
    Connectivity check: Returns if the network is fully connected.
    """
    mg = top.create_nxgraph(net, respect_switches=True)
    g = nx.Graph(mg)
    n_components = nx.number_connected_components(g)
    return n_components == 1

def plot_tree(net):
    """
    Plots the network as a tree, rooted at the slack/ext_grid bus.
    """

    mg = top.create_nxgraph(net, respect_switches=True)
    g = nx.Graph(mg)

    # Build a rooted tree from the slack/ext_grid bus for a clearer hierarchy plot.
    root_bus = int(net.ext_grid.bus.iloc[0])
    tree = nx.bfs_tree(g, root_bus)

    # Layered layout: x by depth, y spread within each depth layer.
    depth = nx.single_source_shortest_path_length(tree, root_bus)
    layers = defaultdict(list)
    for node, d in depth.items():
        layers[d].append(node)

    pos_tree = {}
    for d in sorted(layers.keys()):
        layer = sorted(layers[d])
        if len(layer) == 1:
            ys = [0.0]
        else:
            ys = np.linspace(-1.0, 1.0, len(layer))
        for y, node in zip(ys, layer):
            pos_tree[node] = (d, y)

    plt.figure(figsize=(8, 10))
    nx.draw(
        tree,
        pos_tree,
        node_size=80,
        width=1,
        with_labels=True,
        font_size=6,
        arrows=False,
    )
    plt.gca().invert_xaxis()
    plt.axis("off")
    plt.show()

################################################################

def evaluate_configuration_details(
    net,
    switch_vector,
    voltage_limits=None,
    max_loading_percent=None,
    run_power_flow_only_if_topology_valid=True,
):
    """
    Evaluates the given switch configuration on the provided network and returns detailed metrics about the resulting state of the network, including:
    - cycle_rank: The number of independent cycles in the network, which should be 0 for a radial network.
    - disconnected_components: The number of disconnected components in the network, which should be 0 for a fully connected network.
    - power_flow_failed: A binary indicator of whether the power flow calculation converged successfully (0.0 for success, 1.0 for failure).
    - voltage_violation: The total amount of voltage violation across all buses, calculated as the sum of the magnitudes of any voltage deviations outside the specified limits.
    - line_violation: The total amount of line loading violation across all lines, calculated as the sum of the magnitudes of any line loading percentages that exceed the specified maximum loading percent.
    - loss: The total active power loss in the network, calculated as the sum of the active power losses on all lines after a successful power flow calculation. This is set to infinity if the power flow fails or if the topology is invalid and the power flow is not run.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network object to evaluate.
    switch_vector : list or np.array
        A binary vector indicating the open (False) or closed (True) state of each switch in the network. The length of this vector should match the number of switches in the pandapower network.
    voltage_limits : tuple (vmin, vmax), optional
        A tuple specifying the minimum and maximum voltage limits (in per unit) for the buses. If provided, voltage violations will be calculated based on these limits. If None, voltage violations are not calculated.
    max_loading_percent : float, optional
        A float specifying the maximum allowed loading percentage for the lines. If provided, line loading violations will be calculated based on this limit. If None, line loading violations are not calculated.
    run_power_flow_only_if_topology_valid : bool, optional
        If True, the power flow calculation will only be run if the topology of the network is valid (i.e., cycle_rank == 0 and disconnected_components == 0). If False, the power flow will be attempted regardless of the topology, but any failures will be recorded in the power_flow_failed metric.
    
    Returns
    -------
    dict
        A dictionary containing the following keys and their corresponding values:
        - "cycle_rank": float, the number of independent cycles in the network.
        - "disconnected_components": float, the number of disconnected components in the network.
        - "power_flow_failed": float, 0.0 if power flow converged successfully, 1.0 if it failed.
        - "voltage_violation": float, the total amount of voltage violation across all buses.
        - "line_violation": float, the total amount of line loading violation across all lines.
        - "loss": float, the total active power loss in the network, or infinity if power flow failed or topology is invalid and power flow was not run.
    """
    net.switch["closed"] = np.asarray(switch_vector, dtype=bool)

    mg = top.create_nxgraph(net, respect_switches=True)
    g = nx.Graph(mg)

    n_components = nx.number_connected_components(g)
    cycle_rank = max(0, g.number_of_edges() - g.number_of_nodes() + n_components)

    details = {
        "cycle_rank": float(cycle_rank),
        "disconnected_components": float(max(0, n_components - 1)),
        "power_flow_failed": 0.0,
        "voltage_violation": 0.0,
        "line_violation": 0.0,
        "loss": np.inf,
    }

    topology_is_valid = details["cycle_rank"] == 0 and details["disconnected_components"] == 0
    if run_power_flow_only_if_topology_valid and not topology_is_valid:
        return details

    try:
        pp.runpp(net)
        if not net.converged:
            details["power_flow_failed"] = 1.0
            return details
    except Exception:
        details["power_flow_failed"] = 1.0
        return details

    # if voltage_limits is not None:
    #     vmin, vmax = voltage_limits
    #     vm_pu = net.res_bus.vm_pu.to_numpy()
    #     below_limit = np.clip(vmin - vm_pu, 0.0, None)
    #     above_limit = np.clip(vm_pu - vmax, 0.0, None)
    #     details["voltage_violation"] = float(below_limit.sum() + above_limit.sum())

    # if max_loading_percent is not None:
    #     loading = net.res_line.loading_percent.to_numpy()
    #     overload = np.clip(loading - max_loading_percent, 0.0, None)
    #     details["line_violation"] = float(overload.sum())

    details["loss"] = float(net.res_line.pl_mw.sum())
    return details


def evaluate_configuration(net, switch_vector):
    """
    Evaluate the given network configuration.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network object to evaluate.
    switch_vector : list or np.array
        A binary vector indicating the open (False) or closed (True) state of each switch in the network. The length of this vector should match the number of switches in the pandapower network.

    Returns
    -------
    float
        The total active power loss in the network, or infinity if the configuration is infeasible.
    """

    details = evaluate_configuration_details(
        net,
        switch_vector,
        voltage_limits=None,
        max_loading_percent=None,
    )

    # If all constraints are satisfied, return the loss. Otherwise, return infinity to indicate an infeasible configuration.
    if (
        details["cycle_rank"] > 0
        or details["disconnected_components"] > 0
        or details["power_flow_failed"] > 0
    ):
        return np.inf

    return details["loss"]


DEFAULT_LAGRANGIAN_WEIGHTS = {
    "cycle_rank": 1e3,
    "disconnected_components": 1e3,
    "power_flow_failed": 1e4,
    "voltage_violation": 0.0,
    "line_violation": 0.0,
}


def _lagrangian_penalty(details, penalty_weights=None):
    weights = DEFAULT_LAGRANGIAN_WEIGHTS.copy()
    if penalty_weights is not None:
        weights.update(penalty_weights)

    return sum(weights[name] * details[name] for name in weights)


def _estimate_loss_upper_bound(net):
    total_load_mw = 0.0
    if hasattr(net, "load") and "p_mw" in net.load:
        total_load_mw = float(np.clip(net.load["p_mw"].to_numpy(dtype=float), 0.0, None).sum())

    total_generation_mw = 0.0
    for table_name in ("sgen", "gen"):
        if hasattr(net, table_name):
            table = getattr(net, table_name)
            if "p_mw" in table:
                total_generation_mw += float(np.clip(table["p_mw"].to_numpy(dtype=float), 0.0, None).sum())

    net_demand_mw = max(0.0, total_load_mw - total_generation_mw)
    return max(1.0, net_demand_mw) + 1e-3

################################################################

def dit_representation_to_switch_vector(dit_representation, n_switches):
    switch_vector = [1] * n_switches
    for switch_pos in dit_representation:
        switch_vector[switch_pos] = 0
    return switch_vector

def switch_vector_to_dit_representation(switch_vector):
    return tuple(i for i, state in enumerate(switch_vector) if state == 0)

################################################################

def sample_radial_configuration(number_samples, dit_string_length, dit_dimension, seed=None):
    if dit_string_length > dit_dimension:
        raise ValueError(
            "dit_string_length must be <= dit_dimension when all dits must be different."
        )

    rng = np.random.default_rng(seed)
    samples_dit_strings = [
        DitString(rng.choice(dit_dimension, size=dit_string_length, replace=False), dimension=dit_dimension)
        for _ in range(number_samples)
    ]
    samples_indexes = np.array(
        [s.to_integer() for s in samples_dit_strings],
        dtype=np.int64,
    )

    return samples_indexes, samples_dit_strings

################################################################


def _naive_random_search(
    n_switches,
    n_samples,
    objective_function,
    n_closed_switches=None,
    seed=0,
):
    if n_switches <= 0:
        raise ValueError("n_switches must be strictly positive.")
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative.")

    if n_closed_switches is None:
        n_closed_switches = n_switches
    if not 0 <= n_closed_switches <= n_switches:
        raise ValueError(
            f"n_closed_switches must be between 0 and {n_switches}, got {n_closed_switches}."
        )

    rng = np.random.default_rng(seed)
    best_loss = np.inf
    best_config = None

    for _ in range(n_samples):
        config = np.zeros(n_switches, dtype=bool)
        if n_closed_switches > 0:
            closed_indices = rng.choice(n_switches, size=n_closed_switches, replace=False)
            config[closed_indices] = True

        config = config.tolist()
        loss = objective_function(config)
        if loss < best_loss:
            best_loss = loss
            best_config = config

    return best_loss, best_config

def random_search_min_loss(
    net,
    n_samples,
    seed=0,
    n_closed_switches=None,
    objective_function=None,
):
    n_switches = len(net.switch) if len(net.switch) > 0 else 1
    if n_closed_switches is None:
        n_closed_switches = len(net.bus) - 1 if hasattr(net, "bus") else n_switches
    if objective_function is None:
        objective_function = lambda config: evaluate_configuration(net, config)

    return _naive_random_search(
        n_switches,
        n_samples,
        objective_function=objective_function,
        n_closed_switches=n_closed_switches,
        seed=seed,
    )

################################################################

_bruteforce_net = None


def _bruteforce_init(net_bytes):
    global _bruteforce_net
    import pickle
    _bruteforce_net = pickle.loads(net_bytes)


def _bruteforce_worker(args):
    closed_indices, n_switches = args
    vec = [False] * n_switches
    for i in closed_indices:
        vec[i] = True
    net_copy = copy.deepcopy(_bruteforce_net)
    loss = evaluate_configuration(net_copy, vec)
    return (list(closed_indices), vec, float(loss))


################################################################


class DNR_Network:

    def __init__(self, network_name):
        self.network_name = network_name
        self.network = load_network(network_name)
        self.n_switches = len(self.network.switch)
        self.n_buses = len(self.network.bus)
        self.n_closed_switches = self.n_buses-1 # to have a radial configuration
        self.n_dits = self.n_switches - self.n_closed_switches
        self.default_lagrangian_weights = DEFAULT_LAGRANGIAN_WEIGHTS.copy()
        self.loss_upper_bound = _estimate_loss_upper_bound(self.network)
        self.offset_eval = self.loss_upper_bound

    def _network_with_config(self, config):
        """
        Create a copy of the network with the switch states set according to the provided configuration.
        """

        switch_state = np.asarray(config, dtype=bool)
        if len(switch_state) != self.n_switches:
            raise ValueError(
                f"Expected config of length {self.n_switches}, got {len(switch_state)}"
            )

        configured_net = copy.deepcopy(self.network)
        configured_net.switch["closed"] = switch_state
        return configured_net

    def get_bus(self):
        return self.network.bus
    
    def get_ext_grid(self):
        return self.network.ext_grid
    
    def get_switch(self):
        return self.network.switch
    
    def get_line(self):
        return self.network.line
    
    def dit_representation_to_switch_vector(self, dit_representation):
        return dit_representation_to_switch_vector(dit_representation, self.n_switches)
    
    def switch_vector_to_dit_representation(self, switch_vector):
        return switch_vector_to_dit_representation(switch_vector)

    def sample_radial_configuration(self, number_samples, dit_string_length, dit_dimension, seed=None):
        return sample_radial_configuration(
            number_samples,
            dit_string_length,
            dit_dimension,
            seed=seed,
        )

    @staticmethod
    def sample_fix_ham_weight(number_samples, dit_string_length, dit_dimension, number_of_one, seed=None):
        #Sample state with hamming weight 32
        if seed is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(seed)
        samples_dit_strings = []
        for _ in range(number_samples):
            config = [0] * dit_string_length
            closed_indices = rng.choice(dit_string_length, size=number_of_one, replace=False)
            for idx in closed_indices:
                config[idx] = 1
            samples_dit_strings.append(DitString(config, dimension=dit_dimension))
        samples_indexes = [s.to_integer() for s in samples_dit_strings]
        return samples_indexes, samples_dit_strings 

    def _ensure_geodata(self, net):
        if net.bus["geo"].isna().all():
            # igraph/pygraphviz unavailable; compute layout with built-in networkx
            g = nx.Graph()
            for _, line in net.line.iterrows():
                g.add_edge(int(line.from_bus), int(line.to_bus))
            pos = nx.kamada_kawai_layout(g)
            import json
            for bus_idx, (x, y) in pos.items():
                net.bus.at[bus_idx, "geo"] = json.dumps({"coordinates": [x, y], "type": "Point"})

    def plot_network(self):
        self._ensure_geodata(self.network)
        plot.simple_plot(self.network, respect_switches=False)

    def plot_configuration(self, config):
        configured_network = self._network_with_config(config)
        self._ensure_geodata(configured_network)
        plot.simple_plot(configured_network, respect_switches=True)

    def evaluate(self, config):
        if len(config) == self.n_dits:
            config = self.dit_representation_to_switch_vector(config)
        configured_network = self._network_with_config(config)
        return evaluate_configuration(configured_network, config)

    def evaluate_mcco(self, config):
        objective = self.evaluate(config)
        if not np.isfinite(objective):
            return 0.0
        return float(self.offset_eval) - float(objective)

    def set_lagrangian_penalty_weights(self, penalty_weights):
        self.default_lagrangian_weights.update(penalty_weights)

    def evaluate_lagrangian_mcco(
        self,
        config,
        penalty_weights=None,
        voltage_limits=None,
        max_loading_percent=None,
        return_details=False,
    ):
        if penalty_weights is None:
            penalty_weights = self.default_lagrangian_weights

        if len(config) == self.n_dits:
            config = self.dit_representation_to_switch_vector(config)

        configured_network = self._network_with_config(config)
        details = evaluate_configuration_details(
            configured_network,
            config,
            voltage_limits=voltage_limits,
            max_loading_percent=max_loading_percent,
        )

        base_score = self.evaluate_mcco(config)
        score = base_score - _lagrangian_penalty(details, penalty_weights=penalty_weights)

        if return_details:
            return score, details

        return score

    def analyse_results(
        self,
        results,
        objective_function=None,
        print_results=True,
        representation="dit",
    ):
        if isinstance(results, MatchingPursuitResults):
            solution_positions = results.positions
            dit_strings = results.dit_strings
            values = results.values
        elif isinstance(results, dict):
            solution_positions = results.get("solution_pos", [])
            dit_strings = None
            values = None
        else:
            solution_positions = results
            dit_strings = None
            values = None

        if objective_function is None:
            objective_function = self.evaluate

        if representation not in ("dit", "bit"):
            raise ValueError("representation must be either 'dit' or 'bit'.")

        analysis = []
        for idx, position in enumerate(solution_positions):
            if dit_strings:
                dit_config = dit_strings[idx]
                switch_vector = self.dit_representation_to_switch_vector(dit_config.dit_string)
            elif representation == "dit":
                dit_config = tuple(
                    int(x) for x in DitString.from_integer(position, length=self.n_dits, dimension=self.n_switches)
                )
                switch_vector = self.dit_representation_to_switch_vector(dit_config)
            else:
                switch_vector = tuple(
                    int(x) for x in DitString.from_integer(position, length=self.n_switches, dimension=2)
                )
                dit_config = self.switch_vector_to_dit_representation(switch_vector)

            objective_value = objective_function(switch_vector)
            entry = {
                "solution_pos": int(position),
                "dit_config": dit_config,
                "switch_vector": switch_vector,
                "objective": objective_value,
                "is_radial": self.check_radiality(switch_vector),
                "is_connected": self.check_connectivity(switch_vector),
            }

            if values is not None:
                entry["value"] = values[idx]

            analysis.append(entry)

            if print_results:
                print("solution_pos:", entry["solution_pos"])
                print("dit_config:", entry["dit_config"])
                print("switch_vector:", entry["switch_vector"])
                print("objective:", entry["objective"])
                if "value" in entry:
                    print("value:", entry["value"])
                print()

        return analysis

    def check_connectivity(self, config):
        if len(config) == self.n_dits:
            config = self.dit_representation_to_switch_vector(config)
        configured_network = self._network_with_config(config)
        return check_connectivity(configured_network)

    def check_radiality(self, config):
        if len(config) == self.n_dits:
            config = self.dit_representation_to_switch_vector(config)
        configured_network = self._network_with_config(config)
        return check_radiality(configured_network)

    def naive_random_search_min_loss(
        self,
        n_samples,
        seed=0,
        n_closed_switches=None,
        objective_function=None,
    ):
        if n_closed_switches is None:
            n_closed_switches = self.n_closed_switches
        if objective_function is None:
            objective_function = self.evaluate

        return _naive_random_search(
            self.n_switches,
            n_samples,
            objective_function=objective_function,
            n_closed_switches=n_closed_switches,
            seed=seed,
        )
    
    def plot_tree(self, config):
        if len(config) == self.n_dits:
            config = self.dit_representation_to_switch_vector(config)
        configured_network = self._network_with_config(config)
        plot_tree(configured_network)

    def bruteforce(self, number_of_one=None, n_jobs=None):
        """
        Evaluate all configurations with exactly `number_of_one` closed switches (ones)
        and store the results in a CSV file named <network_name>.csv. Uses all available CPU cores.

        Parameters
        ----------
        number_of_one : int, optional
            Number of closed switches (ones) in each configuration. Defaults to n_closed_switches (radial config).
        n_jobs : int, optional
            Number of parallel worker processes. Defaults to all available CPU cores.

        Returns
        -------
        str
            Path to the written CSV file.
        """
        import csv
        import multiprocessing
        import pickle
        from itertools import combinations


        if n_jobs is None:
            n_jobs = multiprocessing.cpu_count()

        if number_of_one is None:
            number_of_one = self.n_closed_switches
        if not (0 <= number_of_one <= self.n_switches):
            raise ValueError(f"number_of_one must be between 0 and {self.n_switches}")

        all_configs = [
            (combo, self.n_switches)
            for combo in combinations(range(self.n_switches), number_of_one)
        ]

        net_bytes = pickle.dumps(self.network)
        chunksize = max(1, len(all_configs) // (n_jobs * 4))

        with multiprocessing.Pool(
            n_jobs, initializer=_bruteforce_init, initargs=(net_bytes,)
        ) as pool:
            results = pool.map(_bruteforce_worker, all_configs, chunksize=chunksize)

        results.sort(key=lambda x: x[2])

        import os
        output_csv = os.path.join(os.getcwd(), f"{self.network_name}.csv")
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            header = [f"sw_{i}" for i in range(self.n_switches)] + ["loss"]
            writer.writerow(header)
            for _, vec, loss in results:
                row = [int(v) for v in vec] + [loss]
                writer.writerow(row)

    # ------------------------------------------------------------------
    # DNRlib integration
    # ------------------------------------------------------------------

    def _dnrlib_src_path(self):
        return os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', '..', 'DNRlib', 'src', 'DNRlib'
        ))

    def _to_gridcal(self):
        import GridCalEngine.api as gce

        net = self.network
        Zbase = float(net.bus.vn_kv.iloc[0]) ** 2 / float(net.sn_mva)  # ohm

        gc = gce.MultiCircuit()
        gc.Sbase = float(net.sn_mva)

        gc_buses = {}
        for idx, bus in net.bus.iterrows():
            b = gce.Bus(name=f'bus_{idx}', Vnom=float(bus.vn_kv))
            gc.add_bus(b)
            gc_buses[int(idx)] = b

        slack_idx = int(net.ext_grid.bus.iloc[0])
        gc_buses[slack_idx].is_slack = True
        gc.add_external_grid(
            bus=gc_buses[slack_idx],
            api_obj=gce.ExternalGrid(name='slack', Vm=1.0, Va=0.0, mode=gce.ExternalGridMode.VD),
        )

        for i, load in net.load.iterrows():
            gc.add_load(
                gc_buses[int(load.bus)],
                gce.Load(name=f'load_{i}', P=float(load.p_mw), Q=float(load.q_mvar)),
            )

        for i, line in net.line.iterrows():
            r_pu = float(line.r_ohm_per_km * line.length_km) / Zbase
            x_pu = float(line.x_ohm_per_km * line.length_km) / Zbase
            gc.add_line(gce.Line(
                name=f'line_{i}',
                bus_from=gc_buses[int(line.from_bus)],
                bus_to=gc_buses[int(line.to_bus)],
                r=r_pu, x=x_pu,
                active=True,
            ))

        return gc

    def _tie_idtags(self, gc_grid):
        # Tie lines are always the last n_dits entries (added with _tie() last in every network builder)
        return [gc_grid.lines[i].idtag for i in range(self.n_closed_switches, self.n_switches)]

    def _idtags_to_switch_vector(self, gc_grid, disabled_idtags):
        idtag_to_idx = {line.idtag: i for i, line in enumerate(gc_grid.lines)}
        switch_vector = [1] * self.n_switches
        for idtag in disabled_idtags:
            switch_vector[idtag_to_idx[idtag]] = 0
        return switch_vector

    def solve_with_dnrlib(self, method='baran', initial_config=None, **kwargs):
        """
        Solve the DNR problem using a DNRlib algorithm and return the best switch vector.

        Parameters
        ----------
        method : str
            Solver name (case-insensitive). Supported: 'merlin', 'baran', 'salkuti',
            'mstgreedy', 'jakus', 'khalil', 'morton', 'taylor'.
        initial_config : list or array-like, optional
            Starting configuration for local-search methods (baran, salkuti, jakus, morton).
            Accepts either a switch vector of length n_switches or a dit representation of
            length n_dits. If None, defaults to the network's original tie-line configuration.
        **kwargs
            Solver-specific overrides. Sensible defaults are provided for every method:
            - jakus    : PopulationSize=16, MutationProbability=0.4, Niter=20,
                         fitness_ratio=1, loss_factor=0.02
            - khalil   : NumCandidates=10
            - mstgreedy: randomMST=False, algorithm='prim', one=False, current_power=True
            - taylor   : algorithm='SOC', solver='IPOPT', bigM=1e8, Imax=0, vmin=0.9, vmax=1.1

        Returns
        -------
        list[int]
            Binary switch vector (1=closed, 0=open) of length n_switches.
        """
        import contextlib, io
        path = self._dnrlib_src_path()
        if path not in sys.path:
            sys.path.insert(0, path)
        with contextlib.redirect_stdout(io.StringIO()):
            import GridCalEngine.api  # noqa: F401 — suppress the "renamed to VeraGrid" banner
            from GC_DistributionNetworkReconfiguration import DistributionNetworkReconfiguration

        gc_grid = self._to_gridcal()

        if initial_config is not None:
            if len(initial_config) == self.n_dits:
                initial_config = self.dit_representation_to_switch_vector(initial_config)
            idtag_to_idx = {line.idtag: i for i, line in enumerate(gc_grid.lines)}
            idx_to_idtag = {v: k for k, v in idtag_to_idx.items()}
            tie_idtags = [idx_to_idtag[i] for i, s in enumerate(initial_config) if s == 0]
        else:
            tie_idtags = self._tie_idtags(gc_grid)

        defaults = {'TieLines': tie_idtags}
        method_lower = method.lower()
        if method_lower == 'jakus':
            defaults.update(dict(PopulationSize=16, MutationProbability=0.4, Niter=20,
                                 fitness_ratio=1, loss_factor=0.02))
        elif method_lower == 'khalil':
            defaults['NumCandidates'] = 10
        elif method_lower == 'mstgreedy':
            defaults.update(dict(randomMST=False, algorithm='prim', one=False, current_power=True))
        elif method_lower == 'taylor':
            defaults.update(dict(algorithm='SOC', solver='appsi_highs', bigM=1e8, Imax=0, vmin=0.9, vmax=1.1))
        defaults.update(kwargs)

        dnr = DistributionNetworkReconfiguration(grid=gc_grid)
        disabled_idtags = dnr.Solve(method=method, **defaults)
        self.num_pf = dnr.NumPF
        return self._idtags_to_switch_vector(gc_grid, disabled_idtags)
    
    def local_search(self,config):
        """ Perform a local search starting from the given configuration using the 'baran' method in DNRlib, 
            which is a local-search algorithm. The initial configuration can be provided as either a switch vector or a dit representation. 
            The method returns the best switch vector found by the local search. 
            Note that this method may get stuck in local minima, so it is recommended to run it multiple times with different initial configurations for better results.

            Parameters
            ----------
            config : list or array-like
                Starting configuration for the local search. Accepts either a switch vector of length n_switches or a dit representation of length n_dits.
            
            Returns
            -------
            list[int]
                Binary switch vector (1=closed, 0=open) of length n_switches representing the best configuration found by the local search.
        """
        return self.solve_with_dnrlib(method='baran', initial_config=config)

    def solve_via_simulated_annealing(
        self,
        T0: float = 1.0,
        alpha: float = 0.99,
        max_iter: int = 10000,
        patience: int = 100,
        seed=None,
        local_search_fn=None,
        objective_function=None,
    ):
        """
        Solve the DNR problem via simulated annealing.

        Inspired by troma.optimization.classical.simulated_annealing: geometric
        cooling schedule, configurable neighbour proposal, and Metropolis
        acceptance criterion (minimisation).

        Parameters
        ----------
        T0 : float, optional
            Initial temperature. Should be on the order of the expected loss
            difference between neighbours (typically 1e-3 to 1e-1 MW for these
            networks). Default 1.0.
        alpha : float, optional
            Geometric cooling rate in (0, 1). Default 0.99.
        patience : int, optional
            Stop after this many consecutive iterations without any improvement
            to the best solution found so far. Independent of T0 and alpha.
            Default 100.
        seed : int or None, optional
            Random seed for reproducibility.
        local_search_fn : callable, optional
            Neighbour proposal ``f(x: np.ndarray) -> np.ndarray`` that returns
            a new switch vector without modifying the input. Defaults to a swap
            move: open one closed switch and close one open switch, preserving
            the number of closed switches (= n_closed_switches).
        objective_function : callable, optional
            ``f(config: list) -> float`` to minimise. Defaults to
            ``self.evaluate``, which returns total power loss in MW or
            ``np.inf`` for infeasible (non-radial / disconnected) configs.

        Returns
        -------
        best_config : list[int]
            Binary switch vector (1=closed, 0=open) of the best configuration.
        """
        if T0 <= 0:
            raise ValueError("T0 must be > 0.")
        if not (0 < alpha < 1):
            raise ValueError("alpha must be in (0, 1).")
        if patience < 1:
            raise ValueError("patience must be >= 1.")

        rng = np.random.default_rng(seed)

        if objective_function is None:
            objective_function = self.evaluate

        num_pf = 0

        def _counted(config):
            nonlocal num_pf
            num_pf += 1
            return objective_function(config)

        def _default_neighbor(x: np.ndarray) -> np.ndarray:
            x_new = x.copy()
            closed = np.where(x_new == 1)[0]
            opened = np.where(x_new == 0)[0]
            if len(closed) == 0 or len(opened) == 0:
                return x_new
            x_new[int(rng.choice(closed))] = 0
            x_new[int(rng.choice(opened))] = 1
            return x_new

        _propose = local_search_fn if local_search_fn is not None else _default_neighbor

        # Finite sentinel so infeasible (inf) configs can still participate in the
        # Boltzmann factor without producing nan, while being strongly disfavoured.
        _INF_SENTINEL = max(self.loss_upper_bound * 1e6, 1e6)

        def _scalar(v: float) -> float:
            return _INF_SENTINEL if not np.isfinite(v) else float(v)

        # Initial configuration: default radial topology (non-tie switches closed,
        # tie switches open), which is always feasible.
        x = np.array([1] * self.n_closed_switches + [0] * self.n_dits, dtype=int)

        fx = _counted(x.tolist())
        best_x, best_fx = x.copy(), fx
        T = float(T0)
        no_improve = 0

        while (no_improve < patience) and (num_pf < max_iter):
            x_new = np.asarray(_propose(x), dtype=int)
            f_new = _counted(x_new.tolist())

            delta = _scalar(f_new) - _scalar(fx)
            if delta < 0 or rng.random() < np.exp(-delta / T):
                x, fx = x_new, f_new
                if _scalar(fx) < _scalar(best_fx):
                    best_x, best_fx = x.copy(), fx
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1

            T *= alpha

        self.num_pf = num_pf
        return best_x.tolist()

    def evaluate_quality(self, config_or_value, folder=None):
        """
        Given a config (switch vector, DitString, or position list) or a raw objective value,
        return how close this solution is to the global minimum as two percentages:
          - feasible_score: % of feasible (evaluated) configs that are worse than this solution.
          - full_score: % of the entire 2^n_switches space that is worse (unevaluated configs
            are assumed infeasible and counted as worse).
        Both are in [0, 100] where 100 means global minimum, 0 means worst.
        """
        import csv
        import os
        import numpy as np

        if folder is None:
            csv_path = os.path.join(os.getcwd(), f"{self.network_name}.csv")
        else:
            csv_path = os.path.join(os.getcwd(), f"{folder}/{self.network_name}.csv")
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            data = [row for row in reader]

        all_values = np.array([float(row[-1]) for row in data])

        # Resolve the objective value from config or direct float input
        if isinstance(config_or_value, (int, float)):
            value = float(config_or_value)
        else:
            # Convert config to switch vector
            if hasattr(config_or_value, 'dit_string'):
                positions = [int(x) for x in config_or_value.dit_string]
                switch_vector = [1] * self.n_switches
                for pos in positions:
                    switch_vector[pos] = 0
            elif len(config_or_value) == self.n_switches and all(x in (0, 1) for x in config_or_value):
                switch_vector = [int(x) for x in config_or_value]
            else:
                switch_vector = [1] * self.n_switches
                for pos in config_or_value:
                    switch_vector[int(pos)] = 0

            target_str = ''.join(str(x) for x in switch_vector)
            value = None
            for row in data:
                if ''.join(row[:-1]) == target_str:
                    value = float(row[-1])
                    break
            if value is None:
                raise ValueError(f"Configuration {target_str} not found in CSV.")

        # Rank by counting strictly better solutions so any global minimum
        # (including ties) reaches 100%, and the worst solution reaches 0%.
        tol = 1e-12
        n_better = int(np.sum(all_values < (value - tol)))

        feasible_denom = max(len(all_values) - 1, 1)
        feasible_score = (1.0 - n_better / feasible_denom) * 100.0

        total_space = 2 ** self.n_switches
        full_denom = max(total_space - 1, 1)
        # Unevaluated states are assumed infeasible and thus never better.
        full_score = (1.0 - n_better / full_denom) * 100.0
        return feasible_score, full_score