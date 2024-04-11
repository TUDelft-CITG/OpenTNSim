def lag_finder(time_signal, signal1, signal2):
    #Shifts signals in time to maximize the correlation (signals should have the same timestamps)
    nsamples = len(signal2)

    #Smooth signal
    b, a = sc.signal.butter(2, 0.1)
    signal1 = sc.signal.filtfilt(b, a, signal1)
    signal2 = sc.signal.filtfilt(b, a, signal2)

    #Regularize signals by subtracting mean and dividing by standard deviation
    signal2 -= np.mean(signal2);
    signal2 /= np.std(signal2)
    signal1 -= np.mean(signal1);
    signal1 /= np.std(signal1)

    #Find cross-correlation
    xcorr = sc.signal.correlate(signal2, signal1)

    #Create list of tested time shifts
    dt = np.arange(1 - nsamples, nsamples)

    #Determine time step of the signals
    time_step = time_signal[1] - time_signal[0]

    #Determine phase shift in seconds (when correlation is maximum): delay of signal1 with respect to signal2 (so if signal1 is shifted with delay, there is no phase shift between signals)
    delay = dt[xcorr.argmax()] * time_step

    return delay