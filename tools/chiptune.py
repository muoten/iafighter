"""Render work/baute/song.json (melody + chords + beat grid transcribed from the record) as a 16-bit chiptune.
Pulse lead, triangle bass, quiet pulse arpeggio, noise drums. No audio from the record is used.
Out: game/baute_chip.mp3 (starts at song.json t0).
usage: chiptune.py [--lead-only FROM TO OUT.wav]   (just the melody of one passage, times in record seconds)"""
import json, pathlib, subprocess, sys
import numpy as np, soundfile as sf
ROOT = pathlib.Path(__file__).resolve().parent.parent
S = json.load(open(ROOT / 'work/baute/song.json')); T0, T1 = S['t0'], S['t1']; SR = 44100
# ── fit the game's own music (work/original_audio.mp3): its key (E minor -> G major, +1) and its pulse (~129 BPM) ──
KEY = 1; SPEED = 129.2 / 117.45                                   # record time t plays at T0 + (t - T0) / SPEED
json.dump({'key': KEY, 'speed': SPEED}, open(ROOT / 'work/baute/arrange.json', 'w'))
WAVE = np.array(json.load(open(ROOT / 'work/baute/game_wave.json'))['wavetable'])   # one cycle of the game's lead
LEAD_ONLY = '--lead-only' in sys.argv
if LEAD_ONLY: a = sys.argv.index('--lead-only'); T0, T1, OUT = float(sys.argv[a + 1]), float(sys.argv[a + 2]), sys.argv[a + 3]
N = int((T1 - T0 + 1.5) * SR); mix = np.zeros(N)   # sized for record time; trimmed after the speed-up
hz = lambda m: 440 * 2 ** ((m - 69) / 12)
def env(n, a=0.004, r=0.03, sus=0.8):
    e = np.ones(n) * sus; na = max(1, int(a * SR)); nr = min(n, int(r * SR))
    e[:na] = np.linspace(0, 1, na); e[na:na + int(.04 * SR)] = np.linspace(1, sus, len(e[na:na + int(.04 * SR)]))
    if nr: e[-nr:] *= np.linspace(1, 0, nr)
    return e
def pulse(f, n, duty=.25, vib=0.0):
    t = np.arange(n) / SR; ph = np.cumsum(f * (1 + vib * np.sin(2 * np.pi * 5.5 * t) * (t > .12)) / SR)
    return np.where(ph % 1 < duty, 1.0, -1.0)
def wave(f, n, vib=0.0):
    t = np.arange(n) / SR; ph = np.cumsum(f * (1 + vib * np.sin(2 * np.pi * 5.5 * t) * (t > .12)) / SR) % 1
    return np.interp(ph * len(WAVE), np.arange(len(WAVE) + 1), np.r_[WAVE, WAVE[0]])
def tri(f, n): ph = np.arange(n) * f / SR % 1; return 4 * np.abs(ph - .5) - 1
def put(sig, t, g):
    i = int((t - T0) / SPEED * SR); j = min(N, i + len(sig))
    if 0 <= i < N: mix[i:j] += g * sig[:j - i]
for n in [n for n in S['lead'] if T0 <= n['t'] < T1]:                                               # lead, one octave up for the chip feel
    k = int(n['d'] / SPEED * SR * .96); put((.75 * wave(hz(n['p'] + KEY), k, .006) + .25 * pulse(hz(n['p'] + KEY), k, .5)) * env(k, r=.02), n['t'], .15)   # the game's lead register (E3-D#4), sitting in the mix
beats = np.array(S['beats']); bd = np.median(np.diff(beats))
for c in ([] if LEAD_ONLY else S['chords']):
    root = 30 + (c['root'] - 6) % 12 + KEY                               # F#1..F2
    for q in range(4):                                             # bass: root on the 8ths, octave bounce
        k = int(bd / 2 / SPEED * SR * .9); put(tri(hz(root + (12 if q % 2 else 0)), k) * env(k, r=.02, sus=.9), c['t'] + q * bd / 2, .30)
    arp = sorted(54 + (x - 6) % 12 + KEY for x in c['tones'])            # arpeggio on the 16ths, quiet
    for q in range(8):
        k = int(bd / 4 / SPEED * SR * .8); put(pulse(hz(arp[q % 3] + 12), k, .125) * env(k, r=.01, sus=.5), c['t'] + q * bd / 4, .07)
rng = np.random.default_rng(1)
for i, b in enumerate([] if LEAD_ONLY else beats):                                      # drums: kick 1/3, snare 2/4, hats on 8ths
    k = int(.12 * SR); tt = np.arange(k) / SR
    if i % 2 == 0: put(np.sin(2 * np.pi * np.cumsum(150 * np.exp(-tt * 30) + 45) / SR) * np.exp(-tt * 18), b, .45)
    else: put(rng.uniform(-1, 1, k) * np.exp(-tt * 22), b, .22)
    for h in (b, b + bd / 2):
        k = int(.03 * SR); put(rng.uniform(-1, 1, k) * np.exp(-np.arange(k) / SR * 120), h, .06)
mix = mix[:int((T1 - T0 + 1.5) / SPEED * SR)]
mix = np.round(mix / np.abs(mix).max() * .9 * 127) / 127                # 8-bit amplitude steps
def eq_match(x):                                                    # same "speaker" as the CRT recording: match its long-term spectrum
    import librosa
    ref, _ = librosa.load(str(ROOT / 'work/original_audio.mp3'), sr=SR)
    A = np.abs(librosa.stft(ref, n_fft=4096)).mean(1); B = np.abs(librosa.stft(x.astype(np.float32), n_fft=4096)).mean(1) + 1e-9
    g = A / B; f = np.fft.rfftfreq(4096, 1 / SR); sm = np.empty_like(g)
    for i, fi in enumerate(f): k = (f >= fi / 2 ** (1 / 6)) & (f <= fi * 2 ** (1 / 6)); sm[i] = np.exp(np.log(g[k] + 1e-9).mean())   # 1/3-octave smoothing
    sm /= np.median(sm[(f > 300) & (f < 3000)]); sm = np.clip(sm, .05, 4)
    h = np.fft.fftshift(np.fft.irfft(sm)) * np.hanning(4096)
    return np.convolve(x, h, 'same')
if not LEAD_ONLY: mix = eq_match(mix); mix = mix / np.abs(mix).max() * .9
if LEAD_ONLY:
    sf.write(OUT, mix, SR); print('lead only', T0, '-', T1, '->', OUT); sys.exit()
sf.write(ROOT / 'work/baute/chip.wav', mix, SR)
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(ROOT / 'work/baute/chip.wav'), '-af', 'loudnorm=I=-16:TP=-1.5', '-b:a', '128k', str(ROOT / 'game/baute_chip.mp3')], check=True)
print('ok', round(N / SR, 1), 's')
