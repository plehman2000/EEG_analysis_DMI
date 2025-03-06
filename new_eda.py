import marimo

__generated_with = "0.9.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import mne
    import marimo as mo
    # Importing libraries using import keyword.
    import math
    import matplotlib.pyplot as plt
    from scipy import signal
    import marimo as mo
    data_file = mne.io.read_raw_brainvision(r"47.vhdr",preload=True)


    sample_rate = 500



    def get_eeg_section(data_file,start_min, start_sec, end_min, end_sec):
        START_TIME = start_min + (start_sec/60)
        END_TIME = end_min + (end_sec/60)
        # START_TIME = 30.56 # trimeed to jus after hand raise/exhale
        start =int( START_TIME*sample_rate*60)
        stop = int(END_TIME*sample_rate*60)
        # FRUITION_no_self_door = data.filter(l_freq=2, h_freq=80)

        raw = data_file.get_data(start=start, stop=stop)

        return START_TIME, END_TIME, raw





    # Fake Fruition
    fake_fruition_start_time, fake_fruition_end_time, fake_fruition_data = get_eeg_section(data_file, 26, 55, 27, 35)

    real_fruition_start_time, real_fruition_end_time, real_fruition_data = get_eeg_section(data_file, 29, 33, 33, 35)
    #unusual movement at 50s, 200 and 225
     #29 33 start
    # spikes at 30:27, 32:53, 33:18

    real_fruition_start_min  = 30
    real_fruition_start_sec  = 30 
    real_fruition_start_time, real_fruition_end_time, real_fruition_data = get_eeg_section(data_file, real_fruition_start_min, real_fruition_start_sec, 30, 51)



    return (
        data_file,
        fake_fruition_data,
        fake_fruition_end_time,
        fake_fruition_start_time,
        get_eeg_section,
        math,
        mne,
        mo,
        np,
        plt,
        real_fruition_data,
        real_fruition_end_time,
        real_fruition_start_min,
        real_fruition_start_sec,
        real_fruition_start_time,
        sample_rate,
        signal,
    )


@app.cell
def __():
    # HHB or GRFR????
    import emd
    from scipy import ndimage
    return emd, ndimage


@app.cell
def __(
    emd,
    np,
    real_fruition_data,
    real_fruition_start_min,
    real_fruition_start_sec,
    sample_rate,
):
    # import numpy as np
    # import matplotlib.pyplot as plt
    # import emd
    import datetime
    from tqdm import tqdm
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    # Assuming real_fruition_data is a 2D array with shape (26, n_samples)
    # Where 26 is the number of channels and n_samples is the number of time points

    # Extract all 26 channels and apply scaling for graphing
    num_channels = 24

    # Create empty arrays to store IMF and IA from all channels
    all_imfs = []
    all_ias = []

    # Process each channel
    for channel in tqdm(range(num_channels)):
        # Get data for current channel and scale
        real_fruition_data_channel = real_fruition_data[channel] *100 + 1e-8
        
        if np.min(real_fruition_data[channel]) == 0:
            continue

        
        imf = emd.sift.sift(real_fruition_data_channel)
        
        # Calculate instantaneous attributes4
        # print("Calculating...")
        IP, IF, IA = emd.spectra.frequency_transform(imf, sample_rate, 'hilbert')
        
        # Store IMF (using IMF 2 as in original code) and IA for averaging later
        all_imfs.append(imf[:, 2])
        all_ias.append(IA[:, 2])

    # Convert lists to numpy arrays for easier averaging
    all_imfs = np.array(all_imfs)
    all_ias = np.array(all_ias)

    # Calculate average IMF and IA across all channels
    avg_imf = np.mean(all_imfs, axis=0)
    avg_ia = np.mean(all_ias, axis=0)

    # Get data length from averaged IMF
    data_length = avg_imf.shape[0]
    time_indices = np.arange(data_length)

    # Define start time in seconds
    start_time_seconds = real_fruition_start_min * 60 + real_fruition_start_sec

    # Convert sample indices to time in seconds (adding the start time offset)
    times = time_indices / sample_rate + start_time_seconds



    # Function to format time as MM:SS.ss
    def format_time(x, pos):
        minutes = int(x // 60)
        seconds = x % 60
        return f"{minutes:02d}:{seconds:05.2f}"



    return (
        FuncFormatter,
        IA,
        IF,
        IP,
        MultipleLocator,
        all_ias,
        all_imfs,
        avg_ia,
        avg_imf,
        channel,
        data_length,
        datetime,
        format_time,
        imf,
        num_channels,
        real_fruition_data_channel,
        start_time_seconds,
        time_indices,
        times,
        tqdm,
    )


@app.cell
def __(
    FuncFormatter,
    MultipleLocator,
    avg_ia,
    avg_imf,
    format_time,
    plt,
    times,
):
    # Create figure with proper size
    plt.figure(figsize=(12, 4))
    # Plot averaged IMF and IA
    ax1 = plt.subplot2grid((1, 1), (0, 0), rowspan=1)
    ax1.plot(times, avg_imf, 'k', label='Average IMF')
    ax1.plot(times, avg_imf**2, 'k', label='Average IMF Squared')
    ax1.plot(times, avg_ia, 'r', label='Average IA')
    ax1.legend()

    # Calculate the xlim indices based on data length

    ax1.set_title('Average IMF and Instantaneous Amplitude Across 26 Channels')
    ax1.set_xlabel('Time (MM:SS)')
    ax1.set_ylabel('Amplitude')

    # Set x-axis ticks at regular intervals
    loc = MultipleLocator(base=3.0)
    ax1.xaxis.set_major_locator(loc)

    # Set x-axis formatter to show time as MM:SS.ss
    ax1.xaxis.set_major_formatter(FuncFormatter(format_time))

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show plot
    plt.show()
    return ax1, loc


@app.cell
def __(
    FuncFormatter,
    emd,
    np,
    plt,
    real_fruition_data,
    real_fruition_start_min,
    real_fruition_start_sec,
    sample_rate,
    tqdm,
):
    from ssqueezepy import ssq_stft

    def perform_emd_synchrosqueezing(real_fruition_data, sample_rate, real_fruition_start_min, real_fruition_start_sec):
        """
        Perform EMD with Synchrosqueezing analysis on multi-channel data
        
        Parameters:
        -----------
        real_fruition_data : numpy.ndarray
            2D array of channel data (shape: num_channels x num_samples)
        sample_rate : float
            Sampling rate of the data
        real_fruition_start_min : int
            Start time in minutes
        real_fruition_start_sec : float
            Start time in seconds
        
        Returns:
        --------
        dict containing processed data and analysis results
        """
        num_channels = 24
        
        # Initialize lists to store results
        all_synchrosqueezed_imfs = []
        
        # Process each channel
        for channel in tqdm(range(num_channels)):
            # Get data for current channel and scale
            channel_data = real_fruition_data[channel] * 100 + 1e-8
            
            # Skip if channel contains zero values
            if np.min(channel_data) == 0:
                continue
            
            # Perform EMD (Empirical Mode Decomposition)
            imf = emd.sift.sift(channel_data)
            
            # Select a specific IMF (e.g., the third IMF, index 2)
            selected_imf = imf[:, 2]
            
            # Perform Synchrosqueezing on the selected IMF
            # Use STFT-based synchrosqueezing
            Tx, Sx, ssq_freqs, Sfs = ssq_stft(
                selected_imf, 
                fs=sample_rate,  # Sampling frequency
                hop_len=1,       # Minimal hop length for detailed analysis
                squeezing='sum'  # Sum-based squeezing
            )
            
            # Store the synchrosqueezed IMF
            all_synchrosqueezed_imfs.append(Tx)
        
        # Convert to numpy array
        all_synchrosqueezed_imfs = np.array(all_synchrosqueezed_imfs)
        
        # Calculate average of synchrosqueezed IMFs
        avg_synchrosqueezed_imf = np.mean(all_synchrosqueezed_imfs, axis=0)
        summed_synchrosqueezed_imf = np.sum(all_synchrosqueezed_imfs, axis=0)
        
        # Time axis calculation
        data_length = avg_synchrosqueezed_imf.shape[0]
        time_indices = np.arange(data_length)
        start_time_seconds = real_fruition_start_min * 60 + real_fruition_start_sec
        times = time_indices / sample_rate
        
        # Time formatting function
        def format_time(x, pos):
            minutes = int(x // 60)
            seconds = x % 60
            return f"{minutes:02d}:{seconds:05.2f}"
        
        # Visualization
        plt.figure(figsize=(12, 6))
        # plt.imshow(np.abs(summed_synchrosqueezed_imf) - np.mean(np.abs(summed_synchrosqueezed_imf)), aspect='auto', cmap='viridis', origin='lower', vmin=0, vmax=0.1)

        centered_avg = np.abs(avg_synchrosqueezed_imf)
        centered_avg += np.min(centered_avg)
        centered_avg /= np.max(centered_avg)
        
        plt.imshow(centered_avg, aspect='auto', cmap='viridis', origin='lower', vmin=0, vmax=0.8)
        plt.colorbar(label='Magnitude')
        plt.title('Synchrosqueezed IMF Magnitude')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.ylim(0, 50)
        
        # Format x-axis to show time
        ax = plt.gca()
        ax.xaxis.set_major_formatter(FuncFormatter(format_time))
        plt.tight_layout()
        
        return {
            'synchrosqueezed_imfs': all_synchrosqueezed_imfs,
            'avg_synchrosqueezed_imf': avg_synchrosqueezed_imf,
            'times': times,
            'ssq_freqs': ssq_freqs
        }

    # Example usage (commented out)
    results = perform_emd_synchrosqueezing(
        real_fruition_data, 
        sample_rate, 
        real_fruition_start_min, 
        real_fruition_start_sec
    )
    plt.show()
    return perform_emd_synchrosqueezing, results, ssq_stft


@app.cell
def __():

    # # Add HHT and IF in bottom subplot
    # ax2 = plt.subplot2grid((3, 1), (1, 0), rowspan=2)

    # # Create meshgrid for pcolormesh
    # # For pcolormesh, X and Y should be one larger than C in each dimension
    # hht_time_len = hht.shape[1]
    # X = np.linspace(times[0], times[-1], hht_time_len + 1)
    # Y = freq_edges  # freq_edges should already be one larger than the corresponding dimension of hht

    # # Plot HHT with time on x-axis - using proper dimensions
    # im = ax2.pcolormesh(X, Y, hht, cmap='hot_r', vmin=0, shading='flat')

    # # Create grid properly using ax.grid
    # ax2.grid(True, linestyle='-', linewidth=0.3, alpha=0.5, color='gray')

    # # Plot the IF over the HHT with time on x-axis
    # ax2.plot(times, IF[:, 2], 'g', linewidth=3, label='IF')
    # ax2.legend(loc='upper right')

    # # Set limits and labels
    # ax2.set_xlim(times[start_idx], times[end_idx])
    # ax2.set_ylim(8, 20)
    # ax2.set_xlabel('Time (MM:SS)')
    # ax2.set_ylabel('Frequency (Hz)')
    # ax2.set_title('Hilbert-Huang Transform with Instantaneous Frequency')

    # # Set x-axis formatter to show time as MM:SS.ss
    # ax2.xaxis.set_major_formatter(FuncFormatter(format_time))

    # # Add colorbar
    # cb = plt.colorbar(im, ax=ax2)
    # cb.set_label('Amplitude', rotation=90)

    # # Adjust spacing between subplots
    # plt.tight_layout()

    # # Show plot
    # plt.show()
    return


@app.cell
def __():
    return


@app.cell
def __():

    # Define frequency range (low_freq, high_freq, nsteps, spacing)
    # freq_range = (0.1, 10, 80, 'log')
    # f, hht = emd.spectra.hilberthuang(IF, IA, freq_range, sum_time=False)

    # emd.plotting.plot_imfs(imf, time_vect=time_vect)


    return


@app.cell
def __():
    # sample_rate_ex = 1000
    # seconds = 10
    # num_samples = sample_rate_ex*seconds

    # time_vector = np.linspace(0, seconds, num_samples)

    # freq = 5

    # # Change extent of deformation from sinusoidal shape [-1 to 1]
    # nonlinearity_deg = 0.25

    # # Change left-right skew of deformation [-pi to pi]
    # nonlinearity_phi = -np.pi/4

    # # Compute the signal

    # # Create a non-linear oscillation
    # x = emd.simulate.abreu2010(freq, nonlinearity_deg, nonlinearity_phi, sample_rate_ex, seconds)

    # x += np.cos(2 * np.pi * 1 * time_vector)        # Add a simple 1Hz sinusoid
    # x -= np.sin(2 * np.pi * 2.2e-1 * time_vector)   # Add part of a very slow cycle as a trend

    # # Visualise the time-series for analysis
    # plt.figure(figsize=(12, 4))
    # plt.plot(x)

    # imf = emd.sift.sift(x)
    # print(imf.shape)
    # IP, IF, IA = emd.spectra.frequency_transform(imf, sample_rate, 'hilbert')
    # # Define frequency range (low_freq, high_freq, nsteps, spacing)
    # freq_range = (0.1, 10, 80, 'log')
    # f, hht = emd.spectra.hilberthuang(IF, IA, freq_range, sum_time=False)

    # emd.plotting.plot_imfs(imf)
    return


@app.cell
def __():
    return


@app.cell
def __():
    # time_vect.shape = 50000
    # hht_f.shape = 48
    # hht.shape = 48 * 5000
    return


@app.cell
def __(mo):
    mo.md(r"""# Other Stuff""")
    return


@app.cell
def _():
    # import scipy
    # from tqdm import tqdm
    # correlations = {}

    # sh = np.shape(raw)[0]
    # for j in tqdm(range(sh)):
    #     correlations[j] = []
    #     for k in range(sh):
    #         if j != k:
    #             corr = round(np.corrcoef(raw[k], raw[j])[0][1], 5)
    #             if not np.isnan(corr):
    #                 correlations[j].append(corr)
    #             else:
    #                 correlations[j].append(0)

    #     correlations[j] = np.mean(correlations[j])
    return


@app.cell
def _():
    return


@app.cell
def _():
    # def windowed_correlations_detailed(raw, window_size=1000):
    #     """
    #     Calculate windowed correlations between all pairs of time series in raw.

    #     Parameters:
    #     raw (numpy.ndarray): Array of shape (n_series, n_timepoints) containing time series data
    #     window_size (int): Size of the sliding window

    #     Returns:
    #     dict: Dictionary with keys as series indices and values as dictionaries with:
    #          - 'windows': list of window start indices
    #          - 'correlations': dict of correlation values for each other channel in each window
    #     """
    #     n_series = np.shape(raw)[0]
    #     n_timepoints = np.shape(raw)[1]

    #     # Determine how many windows we can have
    #     n_windows = n_timepoints - window_size + 1

    #     # Create windows at regular intervals
    #     windows = list(range(0, n_windows, max(1, min(100, n_windows // 20))))  # Adjust sampling as needed

    #     results = {}

    #     for j in tqdm(range(n_series), desc="Processing channels"):
    #         results[j] = {
    #             'windows': windows,
    #             'correlations': {}
    #         }

    #         for k in range(n_series):
    #             if j != k:
    #                 results[j]['correlations'][k] = []

    #                 for w in windows:
    #                     series1 = raw[j, w:w+window_size]
    #                     series2 = raw[k, w:w+window_size]

    #                     corr = round(np.corrcoef(series1, series2)[0][1], 5)
    #                     if np.isnan(corr):
    #                         corr = 0

    #                     results[j]['correlations'][k].append(corr)

    #     return results

    # def plot_windowed_correlations(results, channel_names=None):
    #     """
    #     Plot the windowed correlations for each channel.

    #     Parameters:
    #     results (dict): Output from windowed_correlations_detailed function
    #     channel_names (list, optional): Names for each channel
    #     """
    #     n_series = len(results)

    #     if channel_names is None:
    #         channel_names = [f"Channel {i}" for i in range(n_series)]

    #     # Create figure with subplots
    #     fig, axes = plt.subplots(n_series, 1, figsize=(12, 3*n_series), sharex=True)
    #     if n_series == 1:
    #         axes = [axes]

    #     # Plot correlations for each channel
    #     for j in range(n_series):
    #         ax = axes[j]
    #         windows = results[j]['windows']

    #         for k, corrs in results[j]['correlations'].items():
    #             ax.plot(windows, corrs, label=f"{channel_names[k]}")

    #         ax.set_title(f"Correlations for {channel_names[j]}")
    #         ax.set_ylabel("Correlation")
    #         ax.set_ylim(-1.05, 1.05)
    #         ax.grid(True, alpha=0.3)

    #         # Only show legend if we have few enough channels for it to be readable
    #         if n_series <= 10:
    #             ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    #     plt.xlabel("Window Start Index")
    #     plt.tight_layout()
    #     plt.show()

    #     # Also create a heatmap visualization
    #     plt.figure(figsize=(12, 8))

    #     # For the heatmap, calculate average correlation between each pair
    #     avg_corr_matrix = np.zeros((n_series, n_series))

    #     for j in range(n_series):
    #         for k in results[j]['correlations']:
    #             avg_corr_matrix[j, k] = np.mean(results[j]['correlations'][k])

    #     plt.imshow(avg_corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    #     plt.colorbar(label="Average Correlation")

    #     plt.xticks(range(n_series), channel_names, rotation=45)
    #     plt.yticks(range(n_series), channel_names)

    #     plt.title("Average Windowed Correlations Between Channels")
    #     plt.tight_layout()
    #     plt.show()
    return


@app.cell
def _():
    # # Example usage:
    # results = windowed_correlations_detailed(raw, window_size=1000)

    # # Optional: Define channel names
    # channel_names = [f"Ch{i}" for i in range(len(results))]

    # # Plot the results
    # plot_windowed_correlations(results, channel_names)
    return


@app.cell
def _():
    # # Calculate the spectrogram using scipy.signal.spectrogram
    # fig, axes = plt.subplots(27, 1, figsize=(12, 50))
    # axes = axes.flatten()
    # spectrograms = []
    # for i in range(26):
    #     ax = axes[i]
    #     # spec, freqs, t, im = ax.specgram(raw[i], Fs=fs, cmap='viridis')
    #     f, t, Zxx = signal.stft(raw[i], fs=fs, nperseg=250, noverlap=125)
    #     # f, t, Zxx = signal.spectrogram(raw[i], fs)
    #     spectrograms.append({"sampled_frequencies":f, "time_segments":t, "STFT_edited":np.abs(Zxx)})
    #     im = ax.pcolormesh(t, f, 10*np.log10(np.abs(Zxx)), shading='gouraud', cmap='viridis')
    #     ax.set_title(f"Channel {i}")
    #     ax.set_xlabel("Time (s)")
    #     ax.set_ylabel("Frequency (Hz)")
    #     # Limit y-axis to show only up to 100 Hz
    #     ax.set_ylim(0, 20)
    #     # ax.plot(raw[i])
    #     # ax.set_title(f"Channel {i}")
    #     # ax.set_xlabel("Time (s)")
    #     # ax.set_ylabel("Amp")
    # plt.tight_layout()
    # plt.show()
    return


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
