"""This module contains utility functions for lock operations in the OpenTNSim simulation environment."""

def _get_lock_operation_to_and_from_node(lock, direction):
    """Get the nodes from and to which the lock operation is directed based on the direction convention

    Convention: direction = 0 (when A -> B), direction = 1 (when B -> A)
        with A = lock.start_node and B = lock.end_node

    Parameters
    ----------
    lock : object
        the lock object generated with IsLockComplex
    direction : int
        the direction of the lock operation: 0 or 1 (see above convention)

    Returns
    -------
    node_of_approach : str
        the name of the node from which the lock operation is directed
    to_node : str
        the name of the node to which the lock operation is directed
    """
    node_of_approach = lock.end_node
    to_node = lock.start_node
    if not direction:
        node_of_approach = lock.start_node
        to_node = lock.end_node
    return node_of_approach, to_node


def _get_lock_object_on_registration_node(multidigraph, registration_node):
    """Get the lock complex object that is associated with a registration node node

    Parameters
    ----------
    multidigraph : nx.MultiDiGraph
        the graph of the simulation as MultiDiGraph-version (to allow for parallel locks between the same node pair)
    registration_node : str
        node name (that has to be in the graph) on which the vessel is currently starting to navigate an edge

    Returns
    -------
    lock : Union(class, None)
        the lock complex object that is associated with the registration node, or None if no lock complex is associated with the registration node
    """
    # check if node is a registration node
    if "Lock_registration_node" not in multidigraph.nodes[registration_node].keys():
        return None

    edge = multidigraph.nodes[registration_node]["Lock_registration_node"]
    # return lock if it exists on the edge
    if "Lock" in multidigraph.edges[edge].keys():
        lock = multidigraph.edges[edge]["Lock"][0]
        return lock
    # Return None if no lock exists on the edge
    else:
        return None


def _update_lock_operation_planning(lock, operation_index, operation_information):
    """Updates the lock operation planning

    Parameters
    ----------
    lock : object
        the lock object generated with IsLockComplex
    operation_index : int
            index of the lock operation
    operation_information : dict
        information to be added to the dataframe with keys as column names and values
    """
    for key, value in operation_information.items():
        if key not in lock.operation_planning.columns:
            warnings.warn(f"Column name ({key}) not in the operation planning dataframe -> skipped.")
            continue
        lock.operation_planning.loc[int(operation_index), key] = value


def _update_lock_vessel_planning(lock, vessel_index, passage_information):
    """Updates the lock vessel planning

    Parameters
    ----------
    lock : object
        the lock object generated with IsLockComplex
    operation_index : int
            index of the lock operation
    passage_information : dict
        information to be added to the dataframe with keys as column names and values
    """
    for key, value in passage_information.items():
        if key not in lock.vessel_planning.columns:
            warnings.warn(f"Column name ({key}) not in the vessel planning dataframe -> skipped.")
            continue
        lock.vessel_planning.loc[vessel_index, key] = value


def determine_route_to_closest_waiting_area(vessel, waiting_area_A, waiting_area_B):
    """
    DOCUMENTATION HERE

    :param node:
    :param vessel:
    :return:
    """

    remaining_route = vessel.route_ahead
    waiting_area_node = None
    for origin in remaining_route:
        if origin == waiting_area_A.edge[0]:
            waiting_area_node = waiting_area_A.edge[1]
            break
        elif origin == waiting_area_B.edge[0]:
            waiting_area_node = waiting_area_B.edge[1]
            break
    if waiting_area_node is not None:
        route_to_waiting_area = vessel.determine_route_to_target_node(target_node=waiting_area_node)
    else:
        route_to_waiting_area = []
        warnings.warn(f"No route found to waiting area")
    return route_to_waiting_area


def _get_lock_operation_direction(lock, to_node):
    """Get the direction of the lock based on the node to which the lock operation is directed

    Convention: direction = 0 (when A -> B), direction = 1 (when B -> A)
        with A = lock.start_node and B = lock.end_node

    Parameters
    ----------
    lock : object
        the lock object generated with IsLockComplex
    to_node : str
        the name of the node to which the lock operation is directed

    Returns
    -------
    direction: int
        the direction of the lock operation: 0 or 1 (see above convention)
    """
    direction = 0
    if to_node == lock.start_node:
        direction = 1

    return direction
