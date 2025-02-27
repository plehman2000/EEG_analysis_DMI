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
    # Calculate the spectrogram using scipy.signal.spectrogram

    fig, axes = plt.subplots(3,9, figsize=(10,16))
    axes = axes.flatten()
    spectrograms = []
    for i in range(26):
        ax = axes[i]
        # spec, freqs, t, im = ax.specgram(raw[i], Fs=fs, cmap='viridis')
        # f,t,Zxx = signal.stft(raw[i], fs=fs, nperseg=250, noverlap=125)
        f, t, Zxx = signal.spectrogram(raw[i], fs)
    
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
def _():
    return


if __name__ == "__main__":
    app.run()
