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

    data = mne.io.read_raw_brainvision("EEG47.vhdr",preload=True)
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
    FRUITION_no_self_door = data.filter(l_freq=2, h_freq=80)

    raw = FRUITION_no_self_door.get_data(start=start, stop=stop)
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
def _(np, raw):
    np.shape(raw)
    return


@app.cell
def _(np, raw):
    import scipy
    from tqdm import tqdm
    correlations = {}

    sh = np.shape(raw)[0]
    for j in tqdm(range(sh)):
        correlations[j] = []
        for k in range(sh):
            if j != k:
                corr = round(np.corrcoef(raw[k], raw[j])[0][1], 5)
                if not np.isnan(corr):
                    correlations[j].append(corr)
                else:
                    correlations[j].append(0)
                
        correlations[j] = np.mean(correlations[j])
    return corr, correlations, j, k, scipy, sh, tqdm


@app.cell
def _(correlations):
    from pprint import pprint

    #TODO run windowed correlatrions
    pprint(correlations)
    return (pprint,)


@app.cell
def _(np, plt, tqdm):


    def windowed_correlations_detailed(raw, window_size=1000):
        """
        Calculate windowed correlations between all pairs of time series in raw.
    
        Parameters:
        raw (numpy.ndarray): Array of shape (n_series, n_timepoints) containing time series data
        window_size (int): Size of the sliding window
    
        Returns:
        dict: Dictionary with keys as series indices and values as dictionaries with:
             - 'windows': list of window start indices
             - 'correlations': dict of correlation values for each other channel in each window
        """
        n_series = np.shape(raw)[0]
        n_timepoints = np.shape(raw)[1]
    
        # Determine how many windows we can have
        n_windows = n_timepoints - window_size + 1
    
        # Create windows at regular intervals
        windows = list(range(0, n_windows, max(1, min(100, n_windows // 20))))  # Adjust sampling as needed
    
        results = {}
    
        for j in tqdm(range(n_series), desc="Processing channels"):
            results[j] = {
                'windows': windows,
                'correlations': {}
            }
        
            for k in range(n_series):
                if j != k:
                    results[j]['correlations'][k] = []
                
                    for w in windows:
                        series1 = raw[j, w:w+window_size]
                        series2 = raw[k, w:w+window_size]
                    
                        corr = round(np.corrcoef(series1, series2)[0][1], 5)
                        if np.isnan(corr):
                            corr = 0
                    
                        results[j]['correlations'][k].append(corr)
    
        return results

    def plot_windowed_correlations(results, channel_names=None):
        """
        Plot the windowed correlations for each channel.
    
        Parameters:
        results (dict): Output from windowed_correlations_detailed function
        channel_names (list, optional): Names for each channel
        """
        n_series = len(results)
    
        if channel_names is None:
            channel_names = [f"Channel {i}" for i in range(n_series)]
    
        # Create figure with subplots
        fig, axes = plt.subplots(n_series, 1, figsize=(12, 3*n_series), sharex=True)
        if n_series == 1:
            axes = [axes]
    
        # Plot correlations for each channel
        for j in range(n_series):
            ax = axes[j]
            windows = results[j]['windows']
        
            for k, corrs in results[j]['correlations'].items():
                ax.plot(windows, corrs, label=f"{channel_names[k]}")
        
            ax.set_title(f"Correlations for {channel_names[j]}")
            ax.set_ylabel("Correlation")
            ax.set_ylim(-1.05, 1.05)
            ax.grid(True, alpha=0.3)
        
            # Only show legend if we have few enough channels for it to be readable
            if n_series <= 10:
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
        plt.xlabel("Window Start Index")
        plt.tight_layout()
        plt.show()
    
        # Also create a heatmap visualization
        plt.figure(figsize=(12, 8))
    
        # For the heatmap, calculate average correlation between each pair
        avg_corr_matrix = np.zeros((n_series, n_series))
    
        for j in range(n_series):
            for k in results[j]['correlations']:
                avg_corr_matrix[j, k] = np.mean(results[j]['correlations'][k])
    
        plt.imshow(avg_corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label="Average Correlation")
    
        plt.xticks(range(n_series), channel_names, rotation=45)
        plt.yticks(range(n_series), channel_names)
    
        plt.title("Average Windowed Correlations Between Channels")
        plt.tight_layout()
        plt.show()


    return plot_windowed_correlations, windowed_correlations_detailed


@app.cell
def _(plot_windowed_correlations, raw, windowed_correlations_detailed):
    # Example usage:
    results = windowed_correlations_detailed(raw, window_size=1000)

    # Optional: Define channel names
    channel_names = [f"Ch{i}" for i in range(len(results))]

    # Plot the results
    plot_windowed_correlations(results, channel_names)

    return channel_names, results


@app.cell
def _(fs, np, plt, raw, signal):
    # Calculate the spectrogram using scipy.signal.spectrogram
    fig, axes = plt.subplots(27, 1, figsize=(12, 50))
    axes = axes.flatten()
    spectrograms = []
    for i in range(26):
        ax = axes[i]
        # spec, freqs, t, im = ax.specgram(raw[i], Fs=fs, cmap='viridis')
        f, t, Zxx = signal.stft(raw[i], fs=fs, nperseg=250, noverlap=125)
        # f, t, Zxx = signal.spectrogram(raw[i], fs)
        spectrograms.append({"sampled_frequencies":f, "time_segments":t, "STFT_edited":np.abs(Zxx)})
        im = ax.pcolormesh(t, f, 10*np.log10(np.abs(Zxx)), shading='gouraud', cmap='viridis')
        ax.set_title(f"Channel {i}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        # Limit y-axis to show only up to 100 Hz
        ax.set_ylim(0, 20)
        # ax.plot(raw[i])
        # ax.set_title(f"Channel {i}")
        # ax.set_xlabel("Time (s)")
        # ax.set_ylabel("Amp")
    plt.tight_layout()
    plt.show()
    return Zxx, ax, axes, f, fig, i, im, spectrograms, t


@app.cell
def _():
    return


@app.cell
def _():
    # # Calculate the spectrogram using scipy.signal.spectrogram

    # fig, axes = plt.subplots(9,3, figsize=(12,36))
    # axes = axes.flatten()
    # spectrograms = []
    # for i in range(26):
    #     ax = axes[i]
    #     # spec, freqs, t, im = ax.specgram(raw[i], Fs=fs, cmap='viridis')
    #     # f,t,Zxx = signal.stft(raw[i], fs=fs, nperseg=250, noverlap=125)
    #     f, t, Zxx = signal.spectrogram(raw[i], fs)

    #     spectrograms.append({"sampled_frequencies":f, "time_segments":t, "STFT_edited":np.abs(Zxx)})
    #     im = ax.pcolormesh(t,f, 10*np.log10(np.abs(Zxx)), shading='gouraud', cmap='viridis')
    #     ax.set_title(f"Channel {i}")
    #     ax.set_xlabel("Time (s)")
    #     ax.set_ylabel("Frequency (Hz)")

    # for i in range(25,27):
    #     axes[i].set_visible(False)

    # plt.tight_layout()

    # plt.show()
    return


if __name__ == "__main__":
    app.run()
