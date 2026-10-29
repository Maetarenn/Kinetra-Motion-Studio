"""Real-time Audio-to-LipSync Fallback Generator.

When the video camera loses sight of the actor's face (actor turns around, looking away, or occluded),
this module analyzes live microphone audio via real-time spectral FFT and energy formant decomposition
to synthesize natural ARKit mouth blendshapes, ensuring continuous dialogue animation.
"""
import math
import threading
import time
from typing import Optional

import numpy as np

from ..core.constants import BLENDSHAPE_NAMES, NBS
from ..core.mathx import OneEuroFilter


class AudioLipSync:
    """Live microphone listener and acoustic-to-phonetic blendshape synthesizer."""

    SAMPLE_RATE = 16000
    BLOCK_SIZE = 512

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._running = False
        self._stream = None
        self._thread = None
        self._lock = threading.Lock()

        # Precompute blendshape name-to-index mapping
        self._idx = {name: i for i, name in enumerate(BLENDSHAPE_NAMES)}

        # Filtered blendshape vector
        self._current_bs = np.zeros(NBS, dtype=np.float32)
        self._filter = OneEuroFilter(f_min=2.5, beta=12.0)
        self._noise_floor = 0.005
        self._gain = 3.8
        self.has_mic = False

    def start(self):
        """Starts asynchronous microphone listening thread."""
        if not self.enabled or self._running:
            return
        self._running = True
        try:
            import sounddevice as sd
            # Test default input device
            dev_info = sd.query_devices(kind="input")
            if dev_info is None:
                self.has_mic = False
                return

            def _callback(indata, frames, time_info, status):
                if not self._running:
                    return
                mono = indata[:, 0]
                self._process_audio_chunk(mono)

            self._stream = sd.InputStream(
                channels=1,
                samplerate=self.SAMPLE_RATE,
                blocksize=self.BLOCK_SIZE,
                callback=_callback,
            )
            self._stream.start()
            self.has_mic = True
        except Exception:
            self.has_mic = False
            self._running = False

    def stop(self):
        """Stops the audio stream."""
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _process_audio_chunk(self, chunk: np.ndarray):
        """Analyzes an audio frame in frequency and time domains."""
        rms = float(np.sqrt(np.mean(chunk ** 2)))

        # Update running noise floor
        if rms < self._noise_floor:
            self._noise_floor = 0.95 * self._noise_floor + 0.05 * rms
        else:
            self._noise_floor = 0.999 * self._noise_floor + 0.001 * rms

        energy = max(0.0, (rms - self._noise_floor) * self._gain)
        energy = min(1.0, energy)

        # FFT Spectrum Analysis
        windowed = chunk * np.hanning(len(chunk))
        fft_vals = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(len(chunk), 1.0 / self.SAMPLE_RATE)

        # Formant bands:
        # Low vowels (A, O, U): 200..750 Hz
        # Mid vowels (E, I): 800..2200 Hz
        # High fricatives (S, T, Z): 2500..6000 Hz
        mask_low = (freqs >= 200) & (freqs <= 750)
        mask_mid = (freqs >= 800) & (freqs <= 2200)
        mask_high = (freqs >= 2500) & (freqs <= 6000)

        e_low = float(np.sum(fft_vals[mask_low]))
        e_mid = float(np.sum(fft_vals[mask_mid]))
        e_high = float(np.sum(fft_vals[mask_high]))
        total_e = e_low + e_mid + e_high + 1e-6

        r_low = e_low / total_e
        r_mid = e_mid / total_e
        r_high = e_high / total_e

        # Synthesize target blendshapes
        target = np.zeros(NBS, dtype=np.float32)

        if energy > 0.04:
            # Jaw Open: strongly driven by overall vocal energy
            jaw = float(np.clip(energy * 1.35 * (0.6 + 0.4 * r_low), 0.0, 1.0))
            target[self._idx["jawOpen"]] = jaw

            # Vowels: A/O/U (Funnel, Pucker)
            funnel = float(np.clip(energy * r_low * 1.2, 0.0, 0.85))
            pucker = float(np.clip(energy * r_low * 0.45, 0.0, 0.6))
            target[self._idx["mouthFunnel"]] = funnel
            target[self._idx["mouthPucker"]] = pucker

            # Vowels: E/I (Smile, Dimple)
            smile = float(np.clip(energy * r_mid * 0.75, 0.0, 0.70))
            target[self._idx["mouthSmileLeft"]] = smile
            target[self._idx["mouthSmileRight"]] = smile
            target[self._idx["mouthLowerDownLeft"]] = float(np.clip(jaw * 0.5, 0.0, 0.6))
            target[self._idx["mouthLowerDownRight"]] = float(np.clip(jaw * 0.5, 0.0, 0.6))

            # Fricatives: S/T (Upper lip rise / shrug)
            shrug = float(np.clip(energy * r_high * 0.4, 0.0, 0.5))
            target[self._idx["mouthShrugUpper"]] = shrug
        else:
            # Silence / Resting mouth
            target[self._idx["mouthClose"]] = 0.05

        with self._lock:
            # Smooth blendshape transitions
            now = time.perf_counter()
            self._current_bs = self._filter(target, now)

    def get_blendshapes(self) -> np.ndarray:
        """Returns the current 52-blendshape array for fallback face tracks."""
        with self._lock:
            return self._current_bs.copy()
