import opentnsim.core


def pass_berth(env, vessel, edge, anchorage_info):
    """Pass an edge that contains a 'Berth' with a vessel. A Berth can be a HasContainer (Quay) or a HasResource (Jetty)."""
    assert (
        "unload_duration" in vessel.metadata
    ), "If you want to unload at a quay a ship should have an unload duration"
    berth = edge["Berth"]

    # For the last edge:
    vessel.log_entry(
        "Start unloading at Berth",
        env.now,
        {"vessel": vessel, "berth": berth},
        berth.geometry,
    )
    # HasContainer -> Quay
    if isinstance(berth, opentnsim.core.HasContainer):
        berth.log_entry(
            "Start unloading at Berth",
            env.now,
            {"vessel": vessel, "berth": berth, "level": berth.container.level},
            berth.geometry,
        )
    # HasResource -> Jetty
    elif isinstance(berth, opentnsim.core.HasResource):
        berth.log_entry(
            "Start unloading at Berth",
            env.now,
            {"vessel": vessel, "berth": berth, "count": berth.resource.count},
            berth.geometry,
        )

    yield vessel.env.timeout(vessel.metadata["unload_duration"])
    vessel.log_entry(
        "Stop unloading at Berth",
        env.now,
        {"vessel": vessel, "berth": berth},
        vessel.geometry,
    )

    if isinstance(berth, opentnsim.core.HasContainer):
        berth.log_entry(
            "Stop unloading at Berth",
            env.now,
            {"vessel": vessel, "berth": berth, "level": berth.container.level},
            berth.geometry,
        )
        # give back space on the Quay
        yield berth.container.put(vessel.length)
        berth.log_entry(
            "Stop release reservation at Berth",
            env.now,
            {"vessel": vessel, "berth": berth, "level": berth.container.level},
            berth.geometry,
        )
    elif isinstance(berth, opentnsim.core.HasResource):
        berth.log_entry(
            "Stop unloading at Berth",
            env.now,
            {"vessel": vessel, "berth": berth, "count": berth.resource.count},
            berth.geometry,
        )
        # release the Jetty
        # we should have requested this spot in the anchorage area, now we can release it.
        berth_req = anchorage_info.get("berth_req")
        assert (
            berth_req is not None
        ), f"Arrived at berth of type resource with berth request {berth_req}\n for vessel \n{vessel} at edge:\n{edge}, with info \n{anchorage_info}"
        yield berth.resource.release(berth_req)
        berth.log_entry(
            "Stop release reservation at Berth",
            env.now,
            {"vessel": vessel, "berth": berth, "count": berth.resource.count},
            berth.geometry,
        )


def pass_anchorage(env, vessel, edge, route, graph, info):
    """Pass an ege with an anchorage. The route argument should be the remaining part of the voyage.
    The info argument should be a dictionary in which we can return some information on requested resources.
    This pass by reference is needed because this is a generator, which means we can not return something and we don't want to change objects inline.

    """
    assert (
        len(route) > 2
    ), "If there is an anchorage in the route, the route should be at least 2 nodes long"

    # TODO: This function is now coupled to berth behaviour.
    # We might want to extract the finding the next berth part and the waiting for an available berth spot by using a callback or something.

    # return a dictionary with information on the requests made in the anchorage area
    next_e = None

    # Some may ships make multiple stops
    # Traverse the route for the first location where you would want to stop
    # Find the next Berth
    for j in range(0, len(route) - 1):
        next_e = (route[j], route[j + 1])
        if "Berth" in graph.edges[next_e]:
            # found a next edge
            break
    else:
        raise ValueError(f"No berth found on route past anchorage {edge}")

    # This is the next stop
    next_edge = graph.edges[next_e]

    # we need to pass this along
    info["next_edge"] = next_edge
    info["out_of_system"] = False

    anchorage = edge["Anchorage"]

    # If there is no space, move out of the system
    if anchorage.resource.count >= anchorage.resource.capacity:
        info["out_of_system"] = True
        vessel.log_entry("Moving out of the system", env.now, "", vessel.geometry)
        return

    # The ship can now request a spot at the anchorage area
    vessel.log_entry("Start wait at Anchorage", env.now, "", vessel.geometry)
    anchorage.log_entry(
        "Request spot at Anchorage",
        env.now,
        {"berth": anchorage, "vessel": vessel, "count": anchorage.resource.count},
        anchorage.geometry,
    )
    # Using this context we will release this request at the end of the with statement
    with anchorage.resource.request() as anchor_req:
        # yielding this will advance time until we have our space at the anchorage
        yield anchor_req
        # Now we are at the anchorage we can request a spot at our target location
        # We know about two types of berths, the one with a container (think quay) an with a resource (thinkjetty)
        assert isinstance(
            next_edge["Berth"],
            (opentnsim.core.HasContainer, opentnsim.core.HasResource),
        )
        # Lookup the berth
        berth = next_edge["Berth"]
        vessel.log_entry(
            "Request spot at Berth",
            env.now,
            {"berth": berth, "vessel": vessel, "length": vessel.length},
            vessel.geometry,
        )
        if isinstance(berth, opentnsim.core.HasContainer):
            # We need information from the ship to be able to moore on a quay
            assert isinstance(
                vessel, opentnsim.core.VesselProperties
            ), "If you have quays (Berths with a container) in your network, all ships should have a length implemented with VesselPropertis"
            yield berth.container.get(vessel.length)
            berth.log_entry(
                "Start assign reservation at Berth",
                env.now,
                {"berth": berth, "vessel": vessel, "level": berth.container.level},
                berth.geometry,
            )
        elif isinstance(berth, opentnsim.core.HasResource):
            berth_req = berth.resource.request()
            # we need to release this request if we are done loading
            info["berth_req"] = berth_req
            yield berth_req
            berth.log_entry(
                "Start assign reservation at Berth",
                env.now,
                {"berth": berth, "vessel": vessel, "count": berth.resource.count},
                berth.geometry,
            )

    anchorage.log_entry(
        "Release spot at Anchorage",
        env.now,
        {"berth": anchorage, "vessel": vessel, "count": anchorage.resource.count},
        anchorage.geometry,
    )
    vessel.log_entry("Leaving Anchorage", env.now, "", vessel.geometry)
