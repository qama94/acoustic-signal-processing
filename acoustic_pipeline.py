"""
NOAA Passive Acoustic Monitoring — Signal Processing Pipeline
=============================================================
Gamar Ismayilova | github.com/qama94/acoustic-signal-processing

Real hydrophone data from NOAA SoundTrap recorder, Monterey Bay
National Marine Sanctuary (MB01), SanctSound programme.

Data source: NOAA Passive Bioacoustic Dataset
https://console.cloud.google.com/storage/browser/noaa-passive-bioacoustic
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import butter, sosfilt, welch, spectrogram
from scipy.ndimage import uniform_filter1d
import soundfile as sf
import warnings
import os
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

SEGMENT_DURATION = 300  # seconds to analyse (5 minutes)
BANDPASS_LOW     = 20   # Hz — lower cutoff
BANDPASS_HIGH    = 8000 # Hz — upper cutoff
DETECTION_WINDOW = 1.0  # seconds — event detection window
DETECTION_THRESH = 6.0  # dB above median — detection threshold

# ─────────────────────────────────────────────────────────────
# 1. LOAD HYDROPHONE DATA (WAV or FLAC)
# ─────────────────────────────────────────────────────────────

def load_hydrophone(filepath, max_seconds=SEGMENT_DURATION):
    """
    Load hydrophone recording — supports WAV and FLAC.
    Reads only first max_seconds to keep processing fast.
    """
    info = sf.info(filepath)
    fs = info.samplerate
    total_samples = info.frames
    total_duration = total_samples / fs
    
    # Read only first max_seconds
    max_samples = min(int(max_seconds * fs), total_samples)
    
    print(f"File info:")
    print(f"  Sample rate : {fs} Hz")
    print(f"  Total duration : {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print(f"  Channels : {info.channels}")
    print(f"  Format : {info.format}")
    print(f"  Loading first {max_seconds}s ({max_samples} samples)...")
    
    data, _ = sf.read(filepath, frames=max_samples, dtype='float64')
    
    # Take first channel if multi-channel
    if data.ndim > 1:
        data = data[:, 0]
    
    # Normalise
    if np.max(np.abs(data)) > 0:
        data = data / np.max(np.abs(data))
    
    print(f"  Loaded: {len(data)} samples | duration: {len(data)/fs:.1f}s")
    return data, fs

# ─────────────────────────────────────────────────────────────
# 2. BUTTERWORTH BANDPASS FILTER
# ─────────────────────────────────────────────────────────────

def bandpass_filter(signal, fs, f_low, f_high, order=4):
    """Apply Butterworth bandpass filter."""
    nyq = fs / 2.0
    f_high = min(f_high, nyq * 0.99)  # stay below Nyquist
    sos = butter(order, [f_low/nyq, f_high/nyq], btype='band', output='sos')
    return sosfilt(sos, signal)

# ─────────────────────────────────────────────────────────────
# 3. POWER SPECTRAL DENSITY
# ─────────────────────────────────────────────────────────────

def compute_psd(signal, fs, nperseg=4096):
    """Estimate PSD using Welch method."""
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg,
                       window='hann', noverlap=nperseg//2)
    psd_db = 10 * np.log10(psd + 1e-20)
    return freqs, psd_db

# ─────────────────────────────────────────────────────────────
# 4. SPECTROGRAM
# ─────────────────────────────────────────────────────────────

def compute_spectrogram(signal, fs, nperseg=1024, noverlap=768):
    """Generate time-frequency spectrogram."""
    freqs, times, Sxx = spectrogram(
        signal, fs=fs, nperseg=nperseg,
        noverlap=noverlap, window='hann'
    )
    Sxx_db = 10 * np.log10(Sxx + 1e-20)
    return times, freqs, Sxx_db

# ─────────────────────────────────────────────────────────────
# 5. EVENT DETECTION
# ─────────────────────────────────────────────────────────────

def detect_events(signal, fs, window_sec=DETECTION_WINDOW,
                  threshold_db=DETECTION_THRESH):
    """Energy-based acoustic event detector."""
    window = int(window_sec * fs)
    hop = window // 2
    n_windows = (len(signal) - window) // hop
    
    energy = np.zeros(n_windows)
    time_axis = np.zeros(n_windows)
    
    for i in range(n_windows):
        start = i * hop
        seg = signal[start:start+window]
        energy[i] = np.sqrt(np.mean(seg**2))
        time_axis[i] = (start + window//2) / fs
    
    energy_db = 20 * np.log10(energy + 1e-10)
    energy_smooth = uniform_filter1d(energy_db, size=5)
    threshold = np.median(energy_smooth) + threshold_db
    
    event_times = []
    in_event = False
    for i, val in enumerate(energy_smooth > threshold):
        if val and not in_event:
            event_times.append(time_axis[i])
            in_event = True
        elif not val:
            in_event = False
    
    return event_times, energy_smooth, time_axis, threshold

# ─────────────────────────────────────────────────────────────
# 6. SNR BY FREQUENCY BAND
# ─────────────────────────────────────────────────────────────

def estimate_snr_bands(signal, fs):
    """Estimate SNR across standard PAM frequency bands."""
    nyq = fs / 2
    bands = {
        'Infrasound\n(1-20 Hz)':   (1, min(20, nyq*0.99)),
        'Low\n(20-200 Hz)':        (20, min(200, nyq*0.99)),
        'Mid\n(200-2000 Hz)':      (200, min(2000, nyq*0.99)),
        'High\n(2-20 kHz)':        (2000, min(20000, nyq*0.99)),
    }
    snr_results = {}
    for name, (fl, fh) in bands.items():
        try:
            filtered = bandpass_filter(signal, fs, fl, fh)
            sig_power = np.mean(filtered**2)
            noise = filtered[:int(min(2, len(signal)/(2*fs))*fs)]
            noise_power = np.mean(noise**2) if len(noise) > 0 else sig_power
            snr = 10 * np.log10((sig_power+1e-20)/(noise_power+1e-20))
            snr_results[name] = round(snr, 1)
        except Exception:
            snr_results[name] = 0.0
    return snr_results

# ─────────────────────────────────────────────────────────────
# 7. FULL PIPELINE
# ─────────────────────────────────────────────────────────────

def run_pipeline(filepath, output_dir='./acoustic_outputs'):
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("NOAA PASSIVE ACOUSTIC MONITORING — SIGNAL PROCESSING")
    print("=" * 60)
    
    # Load
    signal, fs = load_hydrophone(filepath)
    t = np.linspace(0, len(signal)/fs, len(signal))
    
    # Filter
    f_high = min(BANDPASS_HIGH, fs//2 - 100)
    filtered = bandpass_filter(signal, fs, BANDPASS_LOW, f_high)
    print(f"\nBandpass filter: {BANDPASS_LOW}–{f_high} Hz")
    
    # Events
    event_times, energy_db, time_energy, threshold = detect_events(filtered, fs)
    print(f"Detected {len(event_times)} acoustic events:")
    for i, te in enumerate(event_times[:10]):  # show first 10
        print(f"  Event {i+1}: t = {te:.1f}s")
    
    # SNR
    snr_bands = estimate_snr_bands(signal, fs)
    print("\nSNR by frequency band:")
    for band, snr in snr_bands.items():
        print(f"  {band.replace(chr(10),' ')}: {snr} dB")
    
    # ── PLOT 1: Raw vs Filtered ────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    fig.suptitle('NOAA Monterey Bay Hydrophone — Raw vs Filtered Signal',
                 fontsize=13, fontweight='bold')
    
    # Downsample for plotting if signal is long
    ds = max(1, len(signal)//50000)
    
    axes[0].plot(t[::ds], signal[::ds], color='#2C5F8A',
                 linewidth=0.3, alpha=0.8)
    axes[0].set_ylabel('Amplitude (normalised)', fontsize=11)
    axes[0].set_title(f'Raw Signal — fs = {fs} Hz', fontsize=11)
    axes[0].set_xlim([0, t[-1]])
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(t[::ds], filtered[::ds], color='#1A6A3A',
                 linewidth=0.3, alpha=0.8)
    axes[1].set_ylabel('Amplitude (normalised)', fontsize=11)
    axes[1].set_xlabel('Time (s)', fontsize=11)
    axes[1].set_title(f'Bandpass Filtered ({BANDPASS_LOW}–{f_high} Hz)', fontsize=11)
    axes[1].set_xlim([0, t[-1]])
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_raw_vs_filtered.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: 01_raw_vs_filtered.png")
    
    # ── PLOT 2: PSD ────────────────────────────────────────────
    freqs_r, psd_r = compute_psd(signal, fs)
    freqs_f, psd_f = compute_psd(filtered, fs)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(freqs_r, psd_r, color='#2C5F8A', linewidth=1,
            alpha=0.6, label='Raw signal')
    ax.plot(freqs_f, psd_f, color='#1A6A3A', linewidth=1.5,
            label=f'Filtered ({BANDPASS_LOW}–{f_high} Hz)')
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Power Spectral Density (dB)', fontsize=12)
    ax.set_title('Power Spectral Density — Welch Estimate\nNOAA SanctSound MB01, Monterey Bay',
                fontsize=13, fontweight='bold')
    ax.set_xlim([0, min(fs//2, 10000)])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_power_spectral_density.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 02_power_spectral_density.png")
    
    # ── PLOT 3: Spectrogram + Events + SNR ─────────────────────
    # Use shorter segment for spectrogram (first 60s)
    seg_len = min(int(60*fs), len(filtered))
    times_s, freqs_s, Sxx = compute_spectrogram(filtered[:seg_len], fs)
    
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1.2, 0.9], hspace=0.4)
    
    ax1 = fig.add_subplot(gs[0])
    f_plot = min(5000, fs//2)
    fmask = freqs_s <= f_plot
    im = ax1.pcolormesh(times_s, freqs_s[fmask], Sxx[fmask, :],
                        cmap='inferno', shading='gouraud',
                        vmin=np.percentile(Sxx, 10),
                        vmax=np.percentile(Sxx, 99))
    plt.colorbar(im, ax=ax1, label='Power (dB)')
    
    # Mark events that fall within first 60s
    for te in event_times:
        if te <= 60:
            ax1.axvline(te, color='cyan', linewidth=1.5,
                       alpha=0.9, linestyle='--', label='Detected event')
    
    ax1.set_ylabel('Frequency (Hz)', fontsize=11)
    ax1.set_title('Time-Frequency Spectrogram — First 60s\nNOAA SanctSound MB01, Monterey Bay',
                 fontsize=12, fontweight='bold')
    
    # Energy + threshold
    ax2 = fig.add_subplot(gs[1])
    mask_60 = time_energy <= 60
    ax2.plot(time_energy[mask_60], energy_db[mask_60],
             color='#2C5F8A', linewidth=1.2, label='RMS Energy')
    ax2.axhline(threshold, color='red', linestyle='--', linewidth=1.5,
               label=f'Threshold ({threshold:.1f} dB)')
    for te in event_times:
        if te <= 60:
            ax2.axvline(te, color='cyan', linewidth=1.5, alpha=0.8)
    ax2.set_ylabel('Energy (dB)', fontsize=11)
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # SNR bars
    ax3 = fig.add_subplot(gs[2])
    names = list(snr_bands.keys())
    vals = list(snr_bands.values())
    colors = ['#2C5F8A','#1A6A3A','#8A3A1A','#6A1A8A']
    ax3.barh(names, vals, color=colors, alpha=0.8)
    ax3.set_xlabel('SNR (dB)', fontsize=10)
    ax3.set_title('SNR by Frequency Band', fontsize=11, fontweight='bold')
    ax3.axvline(0, color='black', linewidth=0.8)
    ax3.grid(True, alpha=0.3, axis='x')
    
    plt.savefig(f'{output_dir}/03_spectrogram_events_snr.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 03_spectrogram_events_snr.png")
    
    print("\n" + "=" * 60)
    print(f"Pipeline complete.")
    print(f"  Events detected : {len(event_times)}")
    print(f"  Duration analysed : {len(signal)/fs:.1f}s")
    print(f"  Output saved to : {output_dir}")
    print("=" * 60)
    
    return {'n_events': len(event_times), 'event_times': event_times,
            'snr_bands': snr_bands, 'fs': fs}

if __name__ == '__main__':
    import sys
    filepath = sys.argv[1] if len(sys.argv) > 1 else 'hydrophone.flac'
    run_pipeline(filepath)
