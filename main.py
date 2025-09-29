import struct
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import scipy
from scipy import signal

parser = ArgumentParser(description="Convert audio to MakeCode Arcade hex buffers!")
parser.add_argument("-i", "--input", metavar="PATH", type=Path, required=True,
                    help="The input MONO WAV file.")
parser.add_argument("-o", "--output", metavar="PATH", type=Path,
                    required=False,
                    help="The output TypeScript file which contains MakeCode Arcade code.")
parser.add_argument("-p", "--period", metavar="MILLISECONDS", type=int, default=50,
                    help="The period in milliseconds between each DFT for the spectrogram.")
parser.add_argument("--debug", action="store_true",
                    help="Print human readable strings instead of hex buffers for debugging")
args = parser.parse_args()

debug_output = args.debug
can_log = args.output is not None or debug_output
spectrogram_period = args.period

VOLUME_MULTIPLIER = 1.8  # Boost amplitude without losing smoothness

if can_log:
    print(f"Arguments received: {args}")

input_path = args.input.expanduser().resolve()
if can_log:
    print(f"Opening audio {input_path}")

sample_rate, data = scipy.io.wavfile.read(input_path)
channel_count = data.shape[1] if len(data.shape) > 1 else 1
if channel_count > 1:
    print(f"Audio has {channel_count} channels, but only the first will be used.")
sample_count = data.shape[0]
track_length = sample_count / sample_rate

if can_log:
    print(f"Audio has {sample_count} samples at {sample_rate} Hz, "
          f"which is {track_length:.2f} seconds long.")


def constrain(value, min_value, max_value):
    return min(max(value, min_value), max_value)


def create_sound_instruction(start_freq: int, end_freq: int, start_vol: int,
                             end_vol: int, duration: int, waveform: int = 3) -> str:
    """
    Generate a MakeCode Arcade sound instruction.
    waveform: 3=sine, 1=triangle, 2=sawtooth
    """
    return struct.pack("<BBHHHHH",
                       waveform,
                       0,
                       max(start_freq, 1),
                       duration,
                       constrain(start_vol, 0, 1024),
                       constrain(end_vol, 0, 1024),
                       max(end_freq, 1)
                       ).hex()


def audio_to_makecode_arcade(data, sample_rate, period) -> str:
    """
    Convert audio to MakeCode Arcade hex buffers with smoothing and overlap.
    """
    spectrogram_frequency = period / 1000
    nperseg = round(spectrogram_frequency * sample_rate)
    noverlap = nperseg // 2  # 50% overlap for smooth transitions
    if can_log:
        print(f"Generating spectrogram with {period} ms period "
              f"(nperseg={nperseg}, noverlap={noverlap})")

    f, t, Sxx = signal.spectrogram(data, sample_rate, nperseg=nperseg, noverlap=noverlap)

    frequency_buckets = [50, 159, 317, 504, 800, 1270, 2016, 3200, 5080, 9000]
    max_freqs = 20

    if can_log:
        print(f"Gathering {max_freqs} loudest frequencies and amplitudes")

    loudest_indices = np.argsort(Sxx, axis=0)[-max_freqs:]
    loudest_frequencies = f[loudest_indices].transpose()
    loudest_amplitudes = Sxx[loudest_indices, np.arange(Sxx.shape[1])].transpose()
    max_amp = np.max(Sxx)

    # Smooth amplitudes: moving average across slices
    window = 4  # slightly larger for smoother output
    for i in range(loudest_amplitudes.shape[1]):
        loudest_amplitudes[:, i] = np.convolve(loudest_amplitudes[:, i],
                                               np.ones(window)/window,
                                               mode='same')

    if can_log:
        print(f"Generating sound instructions")

    def find_loudest_freq_index_in_bucket(slice_index: int, bucket_index: int) -> int:
        freqs = loudest_frequencies[slice_index]
        low = frequency_buckets[bucket_index - 1] if bucket_index > 0 else 0
        high = frequency_buckets[bucket_index]
        for i in range(len(freqs) - 1, -1, -1):
            if low <= freqs[i] <= high:
                return i
        return -1

    # Build buffers
    sound_instruction_buffers = []
    for bucket_index in range(len(frequency_buckets)):
        buffer = "hex`"
        for slice_index in range(len(loudest_frequencies)):
            freq_index = find_loudest_freq_index_in_bucket(slice_index, bucket_index)
            if freq_index != -1:
                # Average top 2 frequencies for smooth pitch transitions
                top_indices = np.argsort(Sxx[:, slice_index])[-2:]
                freq = round(np.average(loudest_frequencies[slice_index, top_indices],
                                        weights=loudest_amplitudes[slice_index, top_indices]))
                amp = round(loudest_amplitudes[slice_index, freq_index] / max_amp * 1024 * VOLUME_MULTIPLIER)
                amp = min(1024, amp)
                buffer += create_sound_instruction(freq, freq, amp, amp, period)
            else:
                buffer += create_sound_instruction(0, 0, 0, 0, period)
        buffer += "`"
        sound_instruction_buffers.append(buffer)

    # TypeScript output with playInstructions
    ts_code = """namespace music {{
//% shim=music::queuePlayInstructions
export function queuePlayInstructions(timeDelta: number, buf: Buffer) {{}}
}}

const soundInstructions = [
    {}
];

for (const instructions of soundInstructions) {{
    music.playInstructions(100, instructions);
}}""".format(",\n    ".join(sound_instruction_buffers))

    return ts_code


code = audio_to_makecode_arcade(data, sample_rate, spectrogram_period)
if args.output is not None:
    output_path = args.output.expanduser().resolve()
    if can_log:
        print(f"Writing to {output_path}")
    output_path.write_text(code)
else:
    print(code)
