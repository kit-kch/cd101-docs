import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.io.wavfile import write as write_wav
import os

def generate_sine_wave(frequency, duration, sampling_rate=44100, amplitude=0.5):
    """Erzeugt ein Sinussignal."""
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    return t, signal

def generate_combined_sine(frequencies, magnitudes, duration, sampling_rate=44100):
    """
    Erzeugt ein überlagertes Sinussignal aus beliebig vielen Frequenzen und Magnituden
    unter Verwendung der Funktion generate_sine_wave.
    
    Parameters:
        frequencies (list or array): Liste mit Frequenzen [f1, f2, ..., fn]
        magnitudes (list or array): Liste mit Amplituden [a1, a2, ..., an]
        duration (float): Dauer des Signals in Sekunden
        sampling_rate (int): Abtastrate in Hz (Standard: 44100)
    
    Returns:
        t (ndarray): Zeitstempel
        signal (ndarray): Überlagertes Signal
    """
    if len(frequencies) != len(magnitudes):
        raise ValueError("Frequenzen und Magnituden müssen gleich lang sein.")
    
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    combined_signal = np.zeros_like(t)
    
    for freq, mag in zip(frequencies, magnitudes):
        _, sine = generate_sine_wave(freq, duration, sampling_rate, mag)
        combined_signal += sine
    
    return t, combined_signal

def plot_signal(t, signal, filename, xlabel="t", ylabel="A", sample_rate = 44100):
    """Plottet ein Zeitbereichssignal mit Pfeilachsen und angepasstem Stil und speichert als SVG."""

    fig, ax = plt.subplots(figsize=(6, 3))

    # Signal zeichnen
    ax.plot(t, signal, linewidth=2, color='blue')

    # Achsenbeschriftung
    deltaText = 0.0006
    # FIXME: Reduce this, we have 2s now so this needs to be less, was calculated for 5
    if len(t) >= 1 * sample_rate:
        deltaText = 0.35
    ax.text(t[-1] + deltaText, 0, 't / s', fontsize=14, va='center', ha='left', color='black')
    ax.text(0, 1.28, 'A', fontsize=14, va='bottom', ha='center', color='black')

    # Achsenlinien als Pfeile
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    ax.set_yticks([1])
    ax.set_yticklabels(['1'])
    ax.set_ylim(-1.0, 1.2)

    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.tick_params(width=2, direction='inout', length=8)

    ax.grid(False)

    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

    # Speichern
    plt.tight_layout()
    plt.savefig(filename, format="svg", transparent=True)
    plt.close()

def plot_spectrum(frequencies, magnitudes, filename, arrows = True):
    """Plottet manuell ein Spektrum mit Pfeilen an gegebenen Frequenzen und Magnituden."""
    fig, ax = plt.subplots(figsize=(6, 3))

    # Signal zeichnen
    if (arrows):
        for freq, mag in zip(frequencies, magnitudes):
            if (freq < 2000):
                plt.arrow(freq, 0, 0, mag, width = 5, head_width = 30, head_length = 0.1, length_includes_head = True, color='blue')
                plt.text(freq, mag, str(freq), va='bottom', ha='center', fontsize=14, color='blue')
    else:
        ax.plot(frequencies, magnitudes, linewidth=2, color='blue')

    # Achsenbeschriftung
    ax.text(2000 + 50, 0, 'f / Hz', fontsize=14, va='center', ha='left', color='black')
    ax.text(0, 1.28, 'A', fontsize=14, va='bottom', ha='center', color='black')

    # Achsenlinien als Pfeile
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_ylim(-0.1, 1.2)
    ax.set_xlim(-50, 2000)

    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.tick_params(width=2, direction='inout', length=8)

    ax.grid(False)

    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

    plt.tight_layout()
    plt.savefig(filename, format="svg", transparent=True)
    plt.close()


def save_wave_file(filename, signal, sampling_rate = 44100):
    """Speichert ein Signal als WAV-Datei."""
    maxval = max(signal)
    # Scale to 0.90 of max, to leave some headroom
    scaled = ((signal / maxval) * 0.90 * 32767).astype(np.int16)
    write_wav(filename, sampling_rate, scaled)

def sine_plot_wav(frequencies, magnitudes, slide, name, baseF = 0):
    period = 1 / frequencies[0]
    if (baseF):
        period = 1 / baseF

    folder = "out/" + slide + "/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    t, signal = generate_combined_sine(frequencies, magnitudes, 2*period)
    plot_signal(t, signal, folder + name + "_t.svg")
    plot_spectrum(frequencies, magnitudes, folder + name + "_f.svg")

    t, signal = generate_combined_sine(frequencies, magnitudes, 5)
    save_wave_file(folder + name + ".wav", signal)

def generate_lowpass(frequencies, cutoff_frequency):
    frequencies = np.array(frequencies)
    
    # Tiefpass erster Ordnung: |H(f)| = 1 / sqrt(1 + (f/fc)^2)
    attenuation = 1 / np.sqrt(1 + (frequencies / cutoff_frequency)**2)

    return attenuation

def apply_lowpass(frequencies, magnitudes, cutoff_frequency):
    magnitudes = np.array(magnitudes)
    filtered_magnitudes = magnitudes * generate_lowpass(frequencies, cutoff_frequency)
    return filtered_magnitudes

def plot_lowpass(cutoff_frequency, slide, name):
    frequencies = np.arange(0, 2000 + 1, 5)
    magnitudes = generate_lowpass(frequencies, cutoff_frequency)

    folder = "out/" + slide + "/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    plot_spectrum(frequencies, magnitudes, folder + name + "_lp.svg", False)


# Trigger-Signal erzeugen
def generate_beat_trigger(bpm, pattern, shorten_fraction=1/64, total_duration=2.0, sampling_rate=44100, pad=True):
    beat_duration = 60 / bpm
    note_lengths = {
        '16': beat_duration / 4,
        '8': beat_duration / 2,
        '4': beat_duration
    }

    signal = np.array([], dtype=np.float32)
    for note in pattern:
        note_len, note_f = note.split(':')
        duration = note_lengths[note_len]
        active_duration = duration - beat_duration * (shorten_fraction * 4)
        if (note_f == '0'):
            active_duration = 0.0
        rest_duration = duration - active_duration
        active_samples = int(sampling_rate * active_duration)
        rest_samples = int(sampling_rate * rest_duration)
        signal = np.concatenate([signal, np.ones(active_samples), np.zeros(rest_samples)])

    # Auffüllen mit Nullen am Anfang
    if pad:
        total_samples = int(sampling_rate * total_duration)
        pad_samples = total_samples - len(signal)
        if pad_samples > 0:
            signal = np.concatenate([np.zeros(pad_samples), signal])

    t = np.linspace(0, len(signal) / sampling_rate, len(signal), endpoint=False)
    return t, signal

def env_gated_plot_wav(t, signal, ampEnv, slide, name):
    folder = "out/" + slide + "/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    t, beats = generate_beat_trigger(bpm=136, pattern=pattern)
    plot_signal(t, ampEnv, folder + name + "_env.svg")

    plot_signal(t, signal, folder + name + "_t.svg")
    save_wave_file(folder + name + ".wav", signal)

    signal = signal * ampEnv
    plot_signal(t, signal, folder + name + "_mod.svg")
    save_wave_file(folder + name + "_mod.wav", signal)

def generate_multi_signal(signals, pattern, pad=True, duration=2.0, sampling_rate=44100):
    """
    Ersetzt generate_2rect_signal: akzeptiert eine Liste von (frequencies, magnitudes) Paaren.
    pattern benutzt weiterhin '16:1' etc.; note_f wählt die Quelle (1-basiert), '0' = Stille.
    """
    # Erzeuge jede Quellspur
    source_signals = []
    for freqs, mags in signals:
        _, src = generate_combined_sine(freqs, mags, duration, sampling_rate=sampling_rate)
        source_signals.append(src)

    source_signals.insert(0, np.zeros_like(source_signals[0]))

    beat_duration = 60 / 136
    note_lengths = {
        '16': beat_duration / 4,
        '8':  beat_duration / 2,
        '4':  beat_duration,
        '2':  beat_duration * 2,
        '1':  beat_duration * 4
    }

    out = np.array([], dtype=np.float32)
    start = 0
    for note in pattern:
        note_len, note_f = note.split(':')
        duration_sec = note_lengths[note_len]
        end = int(start + int(duration_sec * sampling_rate))

        note_samples = source_signals[int(note_f)]
        out = np.concatenate([out, note_samples[start:end]])
        start = end

    # Auffüllen mit Nullen am Anfang
    if pad:
        total_samples = int(sampling_rate * duration)
        pad_samples = total_samples - len(out)
        if pad_samples > 0:
            out = np.concatenate([source_signals[0][0:pad_samples], out])

    t = np.linspace(0, len(out) / sampling_rate, len(out), endpoint=False)
    return t, out

def gated_plot_wav(frequencies, magnitudes, frequencies2, magnitudes2, pattern, fpattern, slide, name):
    folder = "out/" + slide + "/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    t, beats = generate_beat_trigger(bpm=136, pattern=pattern)
    plot_signal(t, beats, folder + name + "_trig.svg")

    t, signal1 = generate_combined_sine(frequencies, magnitudes, 2)
    t, signal2 = generate_combined_sine(frequencies2, magnitudes2, 2)

    beat_duration = 60 / 136
    sampling_rate = 44100
    note_lengths = {
        '16': beat_duration / 4,
        '8': beat_duration / 2,
        '4': beat_duration,
        '2': beat_duration * 2,
        '1': beat_duration * 4
    }

    signal = np.array([], dtype=np.float32)
    start = 0
    for note in fpattern:
        note_len, note_f = note.split(':')
        duration = note_lengths[note_len]

        note_samples = signal1
        if note_f == '2':
            note_samples = signal2

        end = int(start + int(duration * sampling_rate))
        signal = np.concatenate([signal, note_samples[start : end]])
        start = end

    # Auffüllen mit Nullen am Anfang
    total_samples = int(sampling_rate * 2)
    pad_samples = total_samples - len(signal)
    if pad_samples > 0:
        signal = np.concatenate([signal1[0:pad_samples], signal])

    plot_signal(t, signal, folder + name + "_t.svg")
    save_wave_file(folder + name + ".wav", signal)

    signal = signal * beats
    plot_signal(t, signal, folder + name + "_gated.svg")
    save_wave_file(folder + name + "_gated.wav", signal)

def generate_adsr_envelope(gate, sampling_rate=44100, attack_ms=10, decay_ms=50, sustain_pct=70, release_ms=100, peak=1.0):
    """
    Erzeugt einen ADSR-Envelope als Vektor (0..peak) für ein gegebenes Gate-Signal.
    Gate ist ein Vektor aus 0/1-Werten. Ein steigender Flankenwechsel (0 -> 1) startet
    den Attack/Decay/Sustain-Zyklus, ein fallender (1 -> 0) startet Release.
    
    Parameter:
      gate (array-like): 0/1 Gate (0 oder 1). (steigende Flanke startet Note)
      sampling_rate (int): Abtastrate in Hz
      attack_ms, decay_ms, release_ms (float): Zeiten in Millisekunden
      sustain_pct (float): Sustain-Level in Prozent (0..100)
      peak (float): Maximale Amplitude des Envelopes (Standard 1.0)
    
    Rückgabe:
      envelope (ndarray): Gleiche Länge wie gate, Werte zwischen 0 und peak
    """
    gate = np.asarray(gate).astype(np.int8)
    n = gate.shape[0]
    env = np.zeros(n, dtype=np.float32)

    # Zeiten in Samples
    a_s = max(int((attack_ms / 1000.0) * sampling_rate), 0)
    d_s = max(int((decay_ms / 1000.0) * sampling_rate), 0)
    r_s = max(int((release_ms / 1000.0) * sampling_rate), 0)
    sustain_level = float(np.clip(sustain_pct, 0, 100)) / 100.0 * peak

    # State machine
    state = 'idle'
    idx_in_stage = 0
    value = 0.0

    for i in range(n):
        prev_gate = gate[i - 1] if i > 0 else 0
        g = gate[i]

        # Detect edges
        if prev_gate == 0 and g == 1:
            # retrigger: start attack
            state = 'attack' if a_s > 0 else ('decay' if d_s > 0 else 'sustain')
            idx_in_stage = 0
        elif prev_gate == 1 and g == 0:
            # note off -> release
            state = 'release' if r_s > 0 else 'idle'
            idx_in_stage = 0

        # State behaviours
        if state == 'idle':
            value = 0.0

        elif state == 'attack':
            if a_s == 0:
                value = peak
                state = 'decay' if d_s > 0 else 'sustain'
                idx_in_stage = 0
            else:
                value = (idx_in_stage / a_s) * peak
                idx_in_stage += 1
                if idx_in_stage >= a_s:
                    value = peak
                    state = 'decay' if d_s > 0 else 'sustain'
                    idx_in_stage = 0

        elif state == 'decay':
            if d_s == 0:
                value = sustain_level
                state = 'sustain'
            else:
                # linear from peak -> sustain_level
                value = peak + (sustain_level - peak) * (idx_in_stage / d_s)
                idx_in_stage += 1
                if idx_in_stage >= d_s:
                    value = sustain_level
                    state = 'sustain'

        elif state == 'sustain':
            # remain at sustain while gate is high
            value = sustain_level
            # if gate went high but note shorter than attack+decay, state transitions handled by edge detection above

        elif state == 'release':
            if r_s == 0:
                value = 0.0
                state = 'idle'
            else:
                # linear from current value -> 0 over release samples
                # To keep release consistent even if started from attack/decay, compute start level:
                start_level = value if idx_in_stage == 0 else start_level  # only meaningful when first entering release
                # We set a local start_level on first sample of release
                if idx_in_stage == 0:
                    start_level = value
                value = start_level * (1.0 - (idx_in_stage / r_s))
                idx_in_stage += 1
                if idx_in_stage >= r_s:
                    value = 0.0
                    state = 'idle'
                    idx_in_stage = 0

        env[i] = float(np.clip(value, 0.0, peak))

    return env

def generate_piecewise_amplitude(duration=2.0, sampling_rate=44100):
    """
    Erzeugt ein Signal, das in vier gleichen Vierteln die Amplituden
    [0, 0.25, 0.5, 1] hat. Rückgabe: (t, signal)
    """
    total_samples = int(duration * sampling_rate)
    quarter = total_samples // 4

    # Build quarters
    q1 = np.zeros(quarter, dtype=np.float32)
    q2 = np.full(quarter, 0.25, dtype=np.float32)
    q3 = np.full(quarter, 0.5, dtype=np.float32)
    q4 = np.full(total_samples - 3 * quarter, 1.0, dtype=np.float32)

    signal = np.concatenate([q1, q2, q3, q4])
    t = np.linspace(0, duration, len(signal), endpoint=False)
    return t, signal


def delta_sigma_modulate(signal, sampling_rate=44100, mod_freq=8.0):
    """
    Einfacher erster-Ordnung 1-bit Delta-Sigma Modulator mit block-basierter Ausgabe.

    Die Modulation läuft auf einer niedrigen Ausgabefrequenz `mod_freq` (Hz),
    d.h. jedes ausgegebene Bit bleibt für `1/mod_freq` Sekunden konstant.

    Parameters:
      signal: ndarray der Eingangsamples (erwartet Bereich ~0..1)
      sampling_rate: Sample-Rate der Eingangsamples
      mod_freq: gewünschte Bitrate der 1-bit-Ausgabe (Hz)

    Rückgabe: ndarray gleicher Länge wie `signal` mit Werten 0.0 oder 1.0
    """
    def _clamp(x):
        return float(np.clip(x, 0.0, 1.0))

    sig = np.asarray(signal, dtype=np.float32)
    n = sig.shape[0]
    # sampling_rate and mod_freq are taken from function args

    # If caller supplied attributes on the array (unlikely), ignore — we expect explicit args.

    # Compute block size (samples per output bit)
    block_samples = max(1, int(round(sampling_rate / mod_freq)))

    # Number of blocks
    n_blocks = int(np.ceil(n / block_samples))

    # Compute block-averages as input values for the DSM (keeps same dynamics but enforces bit-hold)
    block_inputs = np.zeros(n_blocks, dtype=np.float32)
    for b in range(n_blocks):
        start = b * block_samples
        end = min(start + block_samples, n)
        if end > start:
            block_inputs[b] = np.mean(sig[start:end])
        else:
            block_inputs[b] = 0.0

    out = np.zeros(n, dtype=np.float32)
    integrator = 0.0
    # start feedback as 1.0 so the first computed output bit becomes 0
    y_prev = 1.0

    for b in range(n_blocks):
        u = _clamp(block_inputs[b])
        # update integrator with block-average input minus previous feedback
        integrator += u - y_prev
        y = 1.0 if integrator >= 0.0 else 0.0
        # fill the corresponding output samples with this bit
        start = b * block_samples
        end = min(start + block_samples, n)
        out[start:end] = y
        y_prev = y

    return out


def apply_time_lowpass(signal, cutoff_freq, sampling_rate=44100):
    """
    Zeitdiskreter Tiefpass 4. Ordnung durch Kaskadierung von vier
    einfachen RC-Erstordnungs-Filtern (exponentielles Gleiten).

    y[n] = y[n-1] + alpha * (x[n] - y[n-1])
    alpha = dt / (RC + dt), RC = 1/(2*pi*fc)

    Diese Implementierung behält die Signatur bei, liefert aber einen
    deutlich steileren Frequenzgang gegenüber einem einzelnen RC-Glätter.
    """
    x = np.asarray(signal, dtype=np.float32)
    if cutoff_freq <= 0:
        return x.copy()

    dt = 1.0 / sampling_rate
    rc = 1.0 / (2.0 * np.pi * float(cutoff_freq))
    alpha = dt / (rc + dt)

    def lp_once(inp):
        out = np.zeros_like(inp)
        if inp.size == 0:
            return out
        out[0] = inp[0]
        for i in range(1, inp.size):
            out[i] = out[i-1] + alpha * (inp[i] - out[i-1])
        return out

    out = x
    # cascade four identical first-order sections -> approx. 4th order
    for _ in range(2):
        out = lp_once(out)

    return out

def slide_1():
    frequencies = [246]
    magnitudes = [1.0]
    sine_plot_wav(frequencies, magnitudes, "slide1", "wave1")

def slide_2():
    frequencies = [246]
    magnitudes = [1.0]
    sine_plot_wav(frequencies, magnitudes, "slide2", "wave1")

    frequencies = [738]
    magnitudes = [1/3]
    sine_plot_wav(frequencies, magnitudes, "slide2", "wave2", baseF = 246)

    frequencies = [246, 738]
    magnitudes = [1.0, 1/3]
    sine_plot_wav(frequencies, magnitudes, "slide2", "wave3")

rect_frequencies = [
    246, 738, 1230, 1722, 2214, 2706, 3198, 3690, 4182, 4674,
    5166, 5658, 6150, 6642, 7134, 7626, 8118, 8610, 9102, 9594,
    10086, 10578, 11070, 11562, 12054, 12546, 13038, 13530, 14022,
    14514, 15006, 15498, 15990, 16482, 16974, 17466, 17958, 18450,
    18942, 19434, 19926
]

rect_magnitudes = [
    1.00000, 0.33333, 0.20000, 0.14286, 0.11111, 0.09091, 0.07692, 0.06667, 0.05882,
    0.05263, 0.04762, 0.04348, 0.04000, 0.03704, 0.03448, 0.03226, 0.03030, 0.02857,
    0.02703, 0.02564, 0.02439, 0.02326, 0.02222, 0.02128, 0.02041, 0.01961, 0.01887,
    0.01818, 0.01754, 0.01695, 0.01639, 0.01587, 0.01538, 0.01493, 0.01449, 0.01408,
    0.01370, 0.01333, 0.01299, 0.01266, 0.01235
]

def slide_3():
    frequencies = [246, 738, 1230]
    magnitudes = [1.0, 1/3, 0.2]
    sine_plot_wav(frequencies, magnitudes, "slide3", "wave1")

    frequencies.append(1722)
    magnitudes.append(0.143)
    sine_plot_wav(frequencies, magnitudes, "slide3", "wave2")


    frequencies = rect_frequencies
    magnitudes = rect_magnitudes
    sine_plot_wav(frequencies, magnitudes, "slide3", "wave3")

def slide_4():
    frequencies = rect_frequencies
    magnitudes = rect_magnitudes
    sine_plot_wav(frequencies, magnitudes, "slide4", "wave1")

    magnitudes = apply_lowpass(frequencies, rect_magnitudes, 500)
    sine_plot_wav(frequencies, magnitudes, "slide4", "wave2")
    plot_lowpass(500, "slide4", "wave2")

def slide_5():
    frequencies = rect_frequencies
    magnitudes = rect_magnitudes
    sine_plot_wav(frequencies, magnitudes, "slide5", "wave1")

    magnitudes = apply_lowpass(frequencies, rect_magnitudes, 4000)
    sine_plot_wav(frequencies, magnitudes, "slide5", "wave2")

    magnitudes = apply_lowpass(frequencies, rect_magnitudes, 2000)
    sine_plot_wav(frequencies, magnitudes, "slide5", "wave3")

    magnitudes = apply_lowpass(frequencies, rect_magnitudes, 1000)
    sine_plot_wav(frequencies, magnitudes, "slide5", "wave4")

    magnitudes = apply_lowpass(frequencies, rect_magnitudes, 500)
    sine_plot_wav(frequencies, magnitudes, "slide5", "wave5")

def slide_6():
    pattern = ['4:1']

    magnitudes = apply_lowpass(rect_frequencies, rect_magnitudes, 500)
    t, signal = generate_combined_sine(rect_frequencies, magnitudes, 2)
    t, env = generate_beat_trigger(bpm=136, pattern=pattern)
    env_gated_plot_wav(t, signal, env, "slide6", "wave1")

pattern = ['16:1', '16:1', '16:1', '16:1', '8:1', '16:1', '16:1', '16:1', '16:1', '16:1', '16:1', '8:1', '16:2', '16:2']

def slide_7():
    magnitudes = apply_lowpass(rect_frequencies, rect_magnitudes, 500)
    t, signal = generate_combined_sine(rect_frequencies, magnitudes, 2)
    t, env = generate_beat_trigger(bpm=136, pattern=pattern)
    env_gated_plot_wav(t, signal, env, "slide7", "wave1")

rect_frequencies2 = [
    330, 990, 1650, 2310, 2970, 3630, 4290, 4950, 5610, 6270,
    6930, 7590, 8250, 8910, 9570, 10230, 10890, 11550, 12210, 12870,
    13530, 14190, 14850, 15510, 16170, 16830, 17490, 18150, 18810, 19470,
]

rect_magnitudes2 = [
    1.0, 0.33333, 0.2, 0.14286, 0.11111, 0.09091, 0.07692, 0.06667, 0.05882, 0.05263,
    0.04762, 0.04348, 0.04, 0.03704, 0.03448, 0.03226, 0.0303, 0.02857, 0.02703, 0.02564,
    0.02439, 0.02326, 0.02222, 0.02128, 0.02041, 0.01961, 0.01887, 0.01818, 0.01754, 0.01695,
]

def slide_8():
    magnitudes = apply_lowpass(rect_frequencies, rect_magnitudes, 500)
    magnitudes2 = apply_lowpass(rect_frequencies2, rect_magnitudes2, 500)
    # use generate_multi_signal with two sources
    t, signal = generate_multi_signal([(rect_frequencies, magnitudes), (rect_frequencies2, magnitudes2)], pattern)
    t, env = generate_beat_trigger(bpm=136, pattern=pattern)
    env_gated_plot_wav(t, signal, env, "slide8", "wave1")


def slide_9():
    pattern = ['4:1', '4:0']

    magnitudes = apply_lowpass(rect_frequencies, rect_magnitudes, 500)
    t, signal = generate_combined_sine(rect_frequencies, magnitudes, 2)
    t, gate = generate_beat_trigger(bpm=136, pattern=pattern)
    env = generate_adsr_envelope(gate, 44100, attack_ms=30, decay_ms=50, sustain_pct=50, release_ms=100, peak=1.0)
    env_gated_plot_wav(t, signal, env, "slide9", "wave1")
    plot_signal(t, gate, 'out/slide9/' + 'wave1' + "_gate.svg")

def slide_10():
    magnitudes = apply_lowpass(rect_frequencies, rect_magnitudes, 500)
    magnitudes2 = apply_lowpass(rect_frequencies2, rect_magnitudes2, 500)
    t, signal = generate_multi_signal([(rect_frequencies, magnitudes), (rect_frequencies2, magnitudes2)], pattern)
    t, gate = generate_beat_trigger(bpm=136, pattern=pattern)
    env = generate_adsr_envelope(gate, 44100, attack_ms=10, decay_ms=20, sustain_pct=70, release_ms=10, peak=1.0)
    env_gated_plot_wav(t, signal, env, "slide10", "wave1")

    env = generate_adsr_envelope(gate, 44100, attack_ms=10, decay_ms=20, sustain_pct=70, release_ms=100, peak=1.0)
    env_gated_plot_wav(t, signal, env, "slide10", "wave2")

    env = generate_adsr_envelope(gate, 44100, attack_ms=10, decay_ms=20, sustain_pct=20, release_ms=10, peak=1.0)
    env_gated_plot_wav(t, signal, env, "slide10", "wave3")

def slide_11():
    # vier Bars wie angegeben
    bar1 = ['16:1', '16:1', '16:1', '16:1', '8:1', '16:1', '16:1', '16:1', '16:1', '16:1', '16:1', '8:1', '16:2', '16:2']
    bar2 = ['16:2', '16:2', '16:2', '16:2', '8:2', '16:3', '16:3', '16:3', '16:3', '16:3', '16:3', '8:3', '16:4', '16:4']
    bar3 = ['16:1', '16:1', '16:1', '16:1', '8:1', '16:1', '16:1', '16:1', '16:1', '16:1', '16:1', '8:1', '16:2', '16:2']
    bar4 = ['16:1', '16:1', '16:1', '16:1', '8:1', '16:1', '16:1', '16:1', '16:1', '16:1', '16:1', '8:1', '16:2', '16:2']

    # Pattern aus den 4 Bars zusammensetzen und zweimal wiederholen
    pattern11 = (bar1 + bar2 + bar3 + bar4)

    # Basistöne: index1 = b3 (246), index2 = e4 (330),
    # index3 = d4 (~293.66), index4 = a3 (220)
    bases = [246.0, 330.0, 293.66, 220.0]

    # BPM / Dauer-Berechnung: eine Bar = 4 Viertel -> total_duration = n_bars * 4 * beat_duration
    bpm = 136
    beat_duration = 60.0 / bpm
    # jede der 4 Bars hat 14 Einträge, pattern11 ist 8 Bars (4*2)
    n_bars = 4
    total_duration = n_bars * 4 * beat_duration  # Sekunden für alle Bars

    def odd_partials_up_to(base, fmax=20000.0):
        freqs = []
        k = 0
        while True:
            f = base * (2 * k + 1)
            if f > fmax:
                break
            freqs.append(f)
            k += 1
        return np.array(freqs)

    cutoff = 500  # wie vorher verwendet

    signals = []
    for base in bases:
        freqs = odd_partials_up_to(base, 20000.0)
        mags = np.array([1.0 / (2 * k + 1) for k in range(len(freqs))])  # 1/(odd harmonic)
        mags = apply_lowpass(freqs, mags, cutoff)
        signals.append((freqs.tolist(), mags.tolist()))

    # Erzeuge Signal und Gate mit korrekter Gesamtdauer
    t, signal = generate_multi_signal(signals, pattern11, pad=False, duration=total_duration, sampling_rate=44100)
    t, gate = generate_beat_trigger(bpm=bpm, pattern=pattern11, total_duration=total_duration, sampling_rate=44100, pad=False)
    env = generate_adsr_envelope(gate, 44100, attack_ms=10, decay_ms=20, sustain_pct=70, release_ms=10, peak=1.0)

    folder = "out/" + 'slide11' + "/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # envelope auf Signallänge anpassen
    if len(signal) > len(env):
        pad_samples = len(signal) - len(env)
        env = np.concatenate([np.zeros(pad_samples), env])
    elif len(env) > len(signal):
        env = env[-len(signal):]

    signal = signal * env

    # ganze Phrase speichern
    save_wave_file(folder + "wave1_mod.wav", signal)



def slide_12():
    """Demo: Erzeuge das piecewise-Signal und ein 1-bit Delta-Sigma-Signal.

    Die Modulationsfrequenz wird auf (4*4)/duration gesetzt (für duration=2s -> 8 Hz).
    """
    sampling_rate = 44100
    duration = 2.0

    # Piecewise amplitude signal (vier Viertel: 0, 0.25, 1, 1)
    t_pw, sig_pw = generate_piecewise_amplitude(duration=duration, sampling_rate=sampling_rate)
    
    # 1-bit Delta-Sigma Modulation applied to the piecewise signal (mod_freq=8 Hz)
    ds = delta_sigma_modulate(sig_pw, sampling_rate=sampling_rate, mod_freq=16.0)

    # Prepare output folder
    folder = "out/slide12/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Plot piecewise and the 1-bit stream (we keep the sine plot optional)
    plot_signal(t_pw, sig_pw, folder + "wave1_t.svg", sample_rate=sampling_rate)
    plot_signal(t_pw, ds, folder + "wave1_ds_t.svg", sample_rate=sampling_rate)

    # Lowpass the 1-bit stream at 4 Hz and save plot (no wav)
    ds_lp = apply_time_lowpass(ds, cutoff_freq=3.0, sampling_rate=sampling_rate)
    plot_signal(t_pw, ds_lp, folder + "wave1_lp.svg", sample_rate=sampling_rate)

slide_1()
slide_2()
slide_3()
slide_4()
slide_5()
slide_6()
slide_7()
slide_8()
slide_9()
slide_10()
slide_11()
slide_12()