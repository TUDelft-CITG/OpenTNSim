class CreateDataset:

    def __init__(self,
                 *args,
                 **kwargs
    ):
        super().__init__(*args, **kwargs)

    def load_hydrodynamic_data(data, dataformat_data, dimformat_data, varformat_data, replacement_data,
                               dataformat_replacement_data, dimformat_replacement_data, varformat_replacement_data,
                               station_interpreter, selected_stations, missing_stations, replacement_nodes,
                               replacement_nodes_use_current_velocity):
        # 1) Load numerical data
        def load_numerical_data(data, dataformat, dimformat, varformat):
            dataset = {}
            water_level_data = []
            easting_current_velocity_data = []
            northing_current_velocity_data = []
            layer_data = []

            def rename_and_transpose_dataarray(data, dims, dimformat, name):
                data.name = name
                for dim in enumerate(data.dims):
                    if dim[1] in data.dims:
                        data = data.rename({dim[1]: dimformat[data.dims[dim[0]]]})

                if data.dims != (dims):
                    data = data.transpose(*dims)

                return data

            for data_set in range(len(data)):
                dataset["dataset{0}".format(data_set)] = data[data_set].rename_vars(varformat)

            for data_set in enumerate(dataset):
                if data_set[0] == 0:
                    if 'Water level' in dataformat:
                        water_level_data = dataset[data_set[1]][dataformat['Water level']]

                    if 'Easting current velocity' in dataformat:
                        easting_current_velocity_data = dataset[data_set[1]][dataformat['Easting current velocity']]

                    if 'Northing current velocity' in dataformat:
                        northing_current_velocity_data = dataset[data_set[1]][dataformat['Northing current velocity']]

                    if 'Layers' in dataformat:
                        layer_data = dataset[data_set[1]][dataformat['Layers']]

                else:
                    if 'Water level' in dataformat:
                        water_level_data = xr.concat(
                            [water_level_data, dataset[data_set[1]][dataformat['Water level']][1:]], "TIME")
                        water_level_data['Stations'] = dataset[data_set[1]][dataformat['Water level']]['Stations']

                    if 'Easting current velocity' in dataformat:
                        easting_current_velocity_data = xr.concat([easting_current_velocity_data, dataset[data_set[1]][
                                                                                                      dataformat[
                                                                                                          'Easting current velocity']][
                                                                                                  1:]], "TIME")
                        easting_current_velocity_data['Stations'] = \
                        dataset[data_set[1]][dataformat['Easting current velocity']]['Stations']

                    if 'Northing current velocity' in dataformat:
                        northing_current_velocity_data = xr.concat([northing_current_velocity_data, dataset[data_set[1]][
                                                                                                        dataformat[
                                                                                                            'Northing current velocity']][
                                                                                                    1:]], "TIME")
                        northing_current_velocity_data['Stations'] = \
                        dataset[data_set[1]][dataformat['Northing current velocity']]['Stations']

                    if 'Layers' in dataformat:
                        layer_data = xr.concat([layer_data, dataset[data_set[1]][dataformat['Layers']][1:]], "TIME")
                        layer_data['Stations'] = dataset[data_set[1]][dataformat['Layers']]['Stations']

            if 'Water level' in dataformat:
                water_level_data = rename_and_transpose_dataarray(water_level_data, ('STATION', 'TIME'), dimformat,
                                                                  'Water level')

            if 'Easting current velocity' in dataformat:
                easting_current_velocity_data = rename_and_transpose_dataarray(easting_current_velocity_data,
                                                                               ('STATION', 'LAYER', 'TIME'), dimformat,
                                                                               'Easting current velocity')

            if 'Northing current velocity' in dataformat:
                northing_current_velocity_data = rename_and_transpose_dataarray(northing_current_velocity_data,
                                                                                ('STATION', 'LAYER', 'TIME'), dimformat,
                                                                                'Northing current velocity')

            if 'Layers' in dataformat:
                layer_data = rename_and_transpose_dataarray(layer_data, ('STATION', 'LAYER', 'TIME'), dimformat, 'Layers')

            return water_level_data, easting_current_velocity_data, northing_current_velocity_data, layer_data

        # 2) Load replacement data
        def load_measurement_data(replacement_data, dataformat_replacement_data, dimformat_replacement_data,
                                  varformat_replacement_data):
            replacement_data = load_numerical_data(replacement_data, dataformat_replacement_data,
                                                   dimformat_replacement_data, varformat_replacement_data)
            return replacement_data

        # 3) Combine datasets
        def combine_dataarrays(data, replacement_data):
            combined_data = []
            for data_set in data:
                combined_data_set = data_set
                for replacement_data_set in replacement_data:
                    if data_set.name == replacement_data_set.name:
                        combined_data_set = xr.concat([data_set, replacement_data_set], 'STATION')
                combined_data.append(combined_data_set)
            return combined_data

            # 4) Clean-up dataset

        def clean_up_dataarrays(combined_data, station_interpreter, selected_stations):
            cleant_combined_data = []
            for data_set in combined_data:
                data_set.transpose()
                data_set = xr.concat([data_set[index] for index in selected_stations], 'STATION')
                data_set['Stations'].values = station_interpreter
                cleant_combined_data.append(data_set)
            return cleant_combined_data

        # 5) Create main dataset
        def create_main_dataset(data):
            dataset = xr.Dataset()
            for data_set in data:
                dataset[data_set.name] = data_set
            return dataset

        # 6) Derive main information based on simple mathematical operations
        def derive_main_information(dataset):
            layer_bounds = []
            layer_bounds.append(dataset['Water level'][0][0].values)
            for layer_depth_center in dataset['Layers'][0].transpose('TIME', 'LAYER')[0].values:
                layer_bounds.append(layer_depth_center - (layer_bounds[-1] - layer_depth_center))

            layer_heights = []
            for bounds in enumerate(layer_bounds):
                if bounds[0] == 0:
                    continue
                layer_heights.append(layer_bounds[bounds[0] - 1] - bounds[1])

            relative_layer_heights = []
            for height in layer_heights:
                relative_layer_heights.append(np.round(height / np.sum(layer_heights), 2))

            dataset = dataset.assign_coords({'Relative layer height': ('LAYER', relative_layer_heights)})
            layer_data = dataset['Layers'].transpose('LAYER', 'STATION', 'TIME')
            dataset['Depth'] = xr.concat([np.mean(
                (dataset['Water level'][index] - layer_data[9][index]) / (1 - dataset['Relative layer height'][-1] / 2) -
                dataset['Water level'][index]) for index in range(len(dataset['Stations']))], 'STATION')
            dataset['Current velocity'] = xr.concat(
                [np.sqrt(dataset['Easting current velocity'][index] ** 2 + dataset['Northing current velocity'][index] ** 2)
                 for index in range(len(dataset['Stations']))], 'STATION')
            dataset['Current direction'] = xr.concat([np.degrees(
                np.arctan2(dataset['Easting current velocity'][index], dataset['Northing current velocity'][index])) for
                                                      index in range(len(dataset['Stations']))], 'STATION')
            astro_water_level_data = []
            for index in range(len(dataset['STATION'])):
                astro_water_level, _, _ = astronomical_tide(dataset['Times'].values, dataset['Water level'][index],
                                                            'Europe/Amsterdam')
                astro_water_level_data.append(astro_water_level[1])
            depth_averaged_current_velocity_data = [np.average(list(dataset['Current velocity'][index].values), axis=0,
                                                               weights=list(dataset['Relative layer height'].values)) for
                                                    index in range(len(dataset['Stations']))]
            depth_averaged_current_direction_data = [np.average(list(dataset['Current direction'][index].values), axis=0,
                                                                weights=list(dataset['Relative layer height'].values)) for
                                                     index in range(len(dataset['Stations']))]
            depth_averaged_easting_current_velocity_data = [
                np.average(list(dataset['Easting current velocity'][index].values), axis=0,
                           weights=list(dataset['Relative layer height'].values)) for index in
                range(len(dataset['Stations']))]
            depth_averaged_northing_current_velocity_data = [
                np.average(list(dataset['Northing current velocity'][index].values), axis=0,
                           weights=list(dataset['Relative layer height'].values)) for index in
                range(len(dataset['Stations']))]
            depth_averaged_data = [astro_water_level_data, depth_averaged_current_velocity_data,
                                   depth_averaged_current_direction_data, depth_averaged_easting_current_velocity_data,
                                   depth_averaged_northing_current_velocity_data]
            depth_averaged_data_name = ['Astronomic water level', 'Depth-averaged current velocity',
                                        'Depth-averaged current direction', 'Depth-averaged easting current velocity',
                                        'Depth-averaged northing current velocity', ]
            for data_set in enumerate(depth_averaged_data):
                data_array = xr.DataArray(data=data_set[1],
                                          dims=["STATION", "TIME"],
                                          coords={'Times': ('TIME', dataset['Times'].values),
                                                  'Stations': ('STATION', dataset['Stations'].values)})
                dataset[depth_averaged_data_name[data_set[0]]] = data_array
            return dataset

        # 7) Derive further information based on complex mathematical operations
        def derive_specific_information(dataset):
            depth_averaged_easting_current_velocity_data = dataset['Depth-averaged easting current velocity'].values
            depth_averaged_northing_current_velocity_data = dataset['Depth-averaged northing current velocity'].values

            depth_averaged_principle_components = [[] for station in range(len(dataset['STATION'].values))]
            for station in range(len(dataset['STATION'].values)):
                depth_averaged_principle_components[station] = fixed2principal_components(
                    depth_averaged_easting_current_velocity_data[station],
                    depth_averaged_northing_current_velocity_data[station])

            depth_averaged_primary_current_velocity = [[] for station in range(len(dataset['STATION'].values))]
            depth_averaged_secondary_current_velocity = [[] for station in range(len(dataset['STATION'].values))]
            for station in range(len(dataset['STATION'].values)):
                depth_averaged_primary_current_velocity[station] = fixed2bearing(
                    depth_averaged_easting_current_velocity_data[station],
                    depth_averaged_northing_current_velocity_data[station],
                    depth_averaged_principle_components[station])
                mean_depth_averaged_primary_current_velocity = np.mean(depth_averaged_primary_current_velocity[station])
                depth_averaged_primary_current_velocity[station] = [value - mean_depth_averaged_primary_current_velocity for
                                                                    value in
                                                                    depth_averaged_primary_current_velocity[station]]

            depth_averaged_primary_current_velocity_data = xr.DataArray(data=depth_averaged_primary_current_velocity,
                                                                        dims=["STATION", "TIME"])

            dataset['Depth-averaged primary current velocity'] = depth_averaged_primary_current_velocity_data

            times_vertical_tidal_periods, times_horizontal_tidal_periods, phase_lags = tidal_periods(dataset,
                                                                                                     'Europe/Amsterdam')

            combined_tidal_periods = [times_vertical_tidal_periods, times_horizontal_tidal_periods]

            for repetition in range(2):
                max_length_tidal_period_data = np.max([len(period) for period in combined_tidal_periods[repetition]])
                tidal_period_data = []
                tidal_periods_data = [[np.zeros(2) for j in np.zeros(max_length_tidal_period_data)] for i in
                                      range(len(dataset['STATION']))]
                for data in tidal_periods_data:
                    empty_list = []
                    for value in data:
                        value[:] = np.nan
                        empty_list.append(value)
                    tidal_period_data.append(empty_list)

                for period in enumerate(combined_tidal_periods[repetition]):
                    for value in enumerate(period[1]):
                        tidal_periods_data[period[0]][value[0]] = value[1]

                combined_tidal_periods[repetition] = tidal_periods_data

            times_vertical_tidal_period_data = xr.DataArray(data=combined_tidal_periods[0],
                                                            dims=["STATION", "VERTTIDES", "TIDEINFO"],
                                                            coords={'Stations': ('STATION', dataset['Stations'].values)})

            times_horizontal_tidal_period_data = xr.DataArray(data=combined_tidal_periods[1],
                                                              dims=["STATION", "HORTIDES", "TIDEINFO"],
                                                              coords={'Stations': ('STATION', dataset['Stations'].values)})

            phase_lag_data = xr.DataArray(data=phase_lags, dims=["STATION"])

            dataset['Vertical tidal periods'] = times_vertical_tidal_period_data
            dataset['Horizontal tidal periods'] = times_horizontal_tidal_period_data
            dataset['Phase lag'] = phase_lag_data
            return dataset

        # 8) Clean-up dataset
        def data_clean_up(dataset):
            dataset = dataset.drop('Easting current velocity')
            dataset = dataset.drop('Northing current velocity')
            dataset = dataset.drop('Depth-averaged easting current velocity')
            dataset = dataset.drop('Depth-averaged northing current velocity')
            dataset = dataset.drop('Depth-averaged current velocity')
            dataset = dataset.drop('Depth-averaged current direction')
            return dataset

        # 9) Expand dataset for missing nodes/measurement stations and separate datasets between nodes and measurement stations
        def expand_data(dataset, missing_stations, replacement_nodes, replacement_nodes_use_current_velocity):
            new_data_array_wlev = dataset['Water level']
            new_data_array_lay = dataset['Layers']
            new_data_array_dep = dataset['Depth']
            new_data_array_curvel = dataset['Current velocity']
            new_data_array_curdir = dataset['Current direction']
            new_data_array_astrowlev = dataset['Astronomic water level']
            new_data_array_verttidalperiods = dataset['Vertical tidal periods']
            new_data_array_hortidalperiods = dataset['Horizontal tidal periods']
            new_data_array_phaselag = dataset['Phase lag']
            new_data_array_daprimcurvel = dataset['Depth-averaged primary current velocity']
            for station in enumerate(replacement_nodes):
                station_index = list(dataset['Stations'].values).index(station[1])
                new_data_wlev = dataset['Water level'][station_index]
                new_data_wlev['Stations'].values = missing_stations[station[0]]
                new_data_lay = dataset['Layers'][station_index]
                new_data_lay['Stations'].values = missing_stations[station[0]]
                new_data_dep = dataset['Depth'][station_index]
                new_data_dep['Stations'].values = missing_stations[station[0]]
                new_data_astrowlev = dataset['Astronomic water level'][station_index]
                new_data_astrowlev['Stations'].values = missing_stations[station[0]]
                new_data_verttidalperiods = dataset['Vertical tidal periods'][station_index]
                new_data_verttidalperiods['Stations'].values = missing_stations[station[0]]
                new_data_hortidalperiods = dataset['Horizontal tidal periods'][station_index]
                new_data_hortidalperiods['Stations'].values = missing_stations[station[0]]
                new_data_phaselag = dataset['Phase lag'][station_index]
                new_data_phaselag['Stations'].values = missing_stations[station[0]]
                new_data_array_wlev = xr.concat([new_data_array_wlev, new_data_wlev], 'STATION')
                new_data_array_lay = xr.concat([new_data_array_lay, new_data_lay], 'STATION')
                new_data_array_dep = xr.concat([new_data_array_dep, new_data_dep], 'STATION')
                new_data_array_astrowlev = xr.concat([new_data_array_astrowlev, new_data_astrowlev], 'STATION')
                new_data_array_verttidalperiods = xr.concat([new_data_array_verttidalperiods, new_data_verttidalperiods],
                                                            'STATION')
                new_data_array_hortidalperiods = xr.concat([new_data_array_hortidalperiods, new_data_hortidalperiods],
                                                           'STATION')
                new_data_array_phaselag = xr.concat([new_data_array_phaselag, new_data_phaselag], 'STATION')
                if replacement_nodes_use_current_velocity[station[0]] == -1:
                    new_data_curvel = dataset['Current velocity'][station_index]
                    new_data_curvel['Stations'].values = missing_stations[station[0]]
                    new_data_curdir = dataset['Current direction'][station_index]
                    new_data_curdir['Stations'].values = missing_stations[station[0]]
                    new_data_daprimcurvel = dataset['Depth-averaged primary current velocity'][station_index]
                    new_data_daprimcurvel['Stations'].values = missing_stations[station[0]]
                    new_data_array_curvel = xr.concat([new_data_array_curvel, new_data_curvel], 'STATION')
                    new_data_array_curdir = xr.concat([new_data_array_curdir, new_data_curdir], 'STATION')
                    new_data_array_daprimcurvel = xr.concat([new_data_array_daprimcurvel, new_data_daprimcurvel], 'STATION')
                else:
                    number_of_layers = len(dataset['Current velocity'][station_index].values)
                    number_of_times = len(dataset['Current velocity'][station_index][0].values)
                    new_data_curvel = dataset['Current velocity'][station_index]
                    new_data_curvel.values = [np.zeros(number_of_times) for layer in range(number_of_layers)]
                    new_data_curvel['Stations'].values = missing_stations[station[0]]
                    new_data_curdir = dataset['Current direction'][station_index]
                    new_data_curdir.values = [np.zeros(number_of_times) for layer in range(number_of_layers)]
                    new_data_curdir['Stations'].values = missing_stations[station[0]]
                    new_data_daprimcurvel = dataset['Depth-averaged primary current velocity'][station_index]
                    new_data_daprimcurvel.values = np.zeros(number_of_times)
                    new_data_daprimcurvel['Stations'].values = missing_stations[station[0]]
                    new_data_array_curvel = xr.concat([new_data_array_curvel, new_data_curvel], 'STATION')
                    new_data_array_curdir = xr.concat([new_data_array_curdir, new_data_curdir], 'STATION')
                    new_data_array_daprimcurvel = xr.concat([new_data_array_daprimcurvel, new_data_daprimcurvel], 'STATION')

            dataset = xr.Dataset()
            dataset['Water level'] = new_data_array_wlev
            dataset['Layers'] = new_data_array_lay
            dataset['Depth'] = new_data_array_dep
            dataset['Current velocity'] = new_data_array_curvel
            dataset['Current direction'] = new_data_array_curdir
            dataset['Astronomic water level'] = new_data_array_astrowlev
            dataset['Vertical tidal periods'] = new_data_array_verttidalperiods
            dataset['Horizontal tidal periods'] = new_data_array_hortidalperiods
            dataset['Phase lag'] = new_data_array_phaselag
            dataset['Depth-averaged primary current velocity'] = new_data_array_daprimcurvel
            return dataset

        data = load_numerical_data(data, dataformat_data, dimformat_data, varformat_data)
        replacement_data = load_measurement_data(replacement_data, dataformat_replacement_data, dimformat_replacement_data,
                                                 varformat_replacement_data)
        data = combine_dataarrays(data, replacement_data)
        data = clean_up_dataarrays(data, station_interpreter, selected_stations)
        data = create_main_dataset(data)
        data = derive_main_information(data)
        data = derive_specific_information(data)
        data = data_clean_up(data)
        data = expand_data(data, missing_stations, replacement_nodes, replacement_nodes_use_current_velocity)
        return data