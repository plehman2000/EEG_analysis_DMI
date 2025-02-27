import marimo

__generated_with = "0.11.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import mne
    # Importing libraries using import keyword.
    import math
    import matplotlib.pyplot as plt
    from scipy import signal

    data = mne.io.read_raw_brainvision("EEG47.vhdr")
    fs = 500

    start_min = 29
    start_sec = 32
    end_min = 33
    end_sec = 40

    START_TIME = start_min + (start_sec/60)

    END_TIME = end_min + (end_sec/60)
    # START_TIME = 30.56 # trimeed to jus after hand raise/exhale
    start =int( START_TIME*fs*60)
    stop = int(END_TIME*fs*60)
    FRUITION_no_self_door = data.get_data(start=start, stop=stop)

    raw = FRUITION_no_self_door


  








    # START_TIME = 25.683 # trimeed to jus after hand raise/exhale
    # start =int( 29.55*fs*60)
    # stop = int(30.95*fs*60)
    # FRUITION_no_self_door = data.get_data(start=start, stop=stop)

    # raw = FRUITION_no_self_door
    return (
        END_TIME,
        FRUITION_no_self_door,
        START_TIME,
        data,
        end_min,
        end_sec,
        fs,
        math,
        mne,
        np,
        plt,
        raw,
        signal,
        start,
        start_min,
        start_sec,
        stop,
    )


@app.cell
def _(fs, np, plt, raw, signal):
    fig, axes = plt.subplots(3,9, figsize=(20,16))
    axes = axes.flatten()
    spectrograms = []
    for i in range(26):
        ax = axes[i]
        # spec, freqs, t, im = ax.specgram(raw[i], Fs=fs, cmap='viridis')
        f,t,Zxx = signal.stft(raw[i], fs=fs, nperseg=250, noverlap=125)
        spectrograms.append({"sampled_frequencies":f, "time_segments":t, "STFT_edited":np.abs(Zxx)})
        im = ax.pcolormesh(t,f, 10*np.log10(np.abs(Zxx)), shading='gouraud', cmap='viridis')
        ax.set_title(f"Channel {i}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

    for i in range(26,27):
        axes[i].set_visible(False)

    plt.tight_layout()

    plt.show()
    return Zxx, ax, axes, f, fig, i, im, spectrograms, t


@app.cell
def _(np):
    from sklearn.metrics.pairwise import cosine_similarity

    def average_cosine_similarities(array_list):

        n = len(array_list)
        if n <= 1:
            return []

        # Ensure all arrays are 1D and convert to 2D for sklearn's cosine_similarity
        arrays_2d = [arr.reshape(1, -1) if arr.ndim == 1 else arr.reshape(1, arr.size) 
                    for arr in array_list]

        avg_similarities = []

        # Calculate full similarity matrix
        full_sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                sim = cosine_similarity(arrays_2d[i], arrays_2d[j])[0][0]
                full_sim_matrix[i, j] = sim
                full_sim_matrix[j, i] = sim  # Symmetric

        # For each array, calculate average similarity with all others
        for i in range(n):
            # Get all similarities for array i (excluding self-similarity which is always 1)
            similarities = full_sim_matrix[i, :]
            # Exclude the similarity with itself (which would be at index i)
            similarities = np.delete(similarities, i)
            # Calculate and store the average
            avg_similarities.append(np.mean(similarities))

        return avg_similarities


    def average_correlation(array_list):
        n = len(array_list)
        if n <= 1:
            return []

        # Ensure all arrays are flattened to 1D
        arrays_1d = [arr.flatten() for arr in array_list]

        average_correlations = []

        # Calculate full similarity matrix
        full_sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                # Calculate correlation between the two 1D arrays
                sim = np.corrcoef(arrays_1d[i], arrays_1d[j])[0, 1]
                full_sim_matrix[i, j] = sim
                full_sim_matrix[j, i] = sim  # Symmetric

        # For each array, calculate average similarity with all others
        for i in range(n):
            # Get all correlations for array i (excluding self-similarity which is always 1)
            correlations = full_sim_matrix[i, :]
            # Exclude the similarity with itself (which would be at index i)
            correlations = np.delete(correlations, i)
            # Calculate and store the average
            average_correlations.append(np.mean(correlations))

        return average_correlations
    return average_correlation, average_cosine_similarities, cosine_similarity


@app.cell
def _(average_correlation, np, spectrograms):
    stfts = [x['STFT_edited'] for x in spectrograms] # d1 is freq intensity, d2 is time

    #calc cosine sim across each stft across time
    print(
    np.shape(stfts)
    )
    correlations = []

    similarities = []
    for time_idx in range(np.shape(stfts[0])[1]):
        # print(time_idx)

        slices = [stft_sample[:,time_idx] for  stft_sample in stfts]
        # print(slices)
        correlations.append(average_correlation(slices))

    correlations = np.swapaxes(correlations,0,1)
    return correlations, similarities, slices, stfts, time_idx


@app.cell
def _(START_TIME, correlations, np, plt, t):
    import datetime
    import matplotlib.dates as mdates

    # Assuming these variables are defined elsewhere in your code
    # Define them here for the example

    plt.figure(figsize=(12, 5))

    # Use meshgrid to create proper x and y coordinates for pcolormesh
    T = np.meshgrid([(x/60)+START_TIME for x in t])

    # Plot the heatmap using the coordinates and correlations data (first 24 channels)
    plt.pcolormesh(T, [f"{x+1}" for x in np.arange(20,24)], correlations[20:24], cmap='viridis')

    # Create proper time formatting for x-axis
    time_ticks_pos = [(x/60)+START_TIME for x in t[::6]]  # Use fewer ticks for readability
    time_ticks_labels = []

    for duration in time_ticks_pos:
        minutes = int(duration)
        seconds = int((duration - minutes) * 60)
        time_tick = f"{minutes}:{seconds:02d}"
        time_ticks_labels.append(time_tick)

    # Set the ticks and labels
    plt.xticks(time_ticks_pos, time_ticks_labels)

    # Add colorbar and labels
    plt.colorbar(label='Correlation')
    plt.title("Average of (One Channel to All Other Channel) Correlations of STFT of EEG")
    plt.xlabel("Time (min:sec)")
    plt.ylabel("Channels (trimmed 25,26 due to uninformativeness)")
    plt.tight_layout()
    plt.show()
    return (
        T,
        datetime,
        duration,
        mdates,
        minutes,
        seconds,
        time_tick,
        time_ticks_labels,
        time_ticks_pos,
    )


if __name__ == "__main__":
    app.run()
