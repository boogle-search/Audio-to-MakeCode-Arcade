import struct
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import scipy

parser = ArgumentParser(description="Convert audio to MakeCode Arcade hex buffers!")
parser.add_argument("-i", "--input", metavar="PATH", type=Path, required=True,
                    help="The input MONO WAV file.")
parser.add_argument("-o", "--output", metavar="PATH", type=Path,
                    required=False,
                    help="The output TypeScript file which contains MakeCode Arcade code.")
parser.add_argument("-p", "--period", metavar="MILLISECONDS", type=int, default=25,
                    help="The period in milliseconds between each DFT for the spectrogram.")
parser.add_argument("--debug", action="store_true",
                    help="Print human readable strings instead of hex buffers for debugging")
args = parser.parse_args()

debug_output = args.debug
can_log = args.output is not None or debug_output
spectrogram_period = args.period

if can_log:
    print(f"Arguments received: {args}")

input_path = args.input.expanduser().resolve()
if can_log:
    print(f"Opening audio {input_path}")

# Read WAV file
sample_rate, data = scipy.io.wavfile.read(input_path)

# Convert to mono if necessary
if len(data.shape) > 1 and data.shape[1] > 1:
    print(f"Audio has {data.shape[1]} channels, using only the first channel.")
    data = data[:, 0]

sample_count = data.shape[0]
track_length = sample_count / sample_rate
if can_log:
    print(f"Audio has {sample_count} samples at {sample_rate} Hz, {track_length:.2f} seconds long.")


def constrain(value, min_value, max_value):
    return min(max(value, min_value), max_value)


def create_sound_instruction(start_freq: int, end_freq: int, start_vol: int,
                             end_vol: int, duration: int) -> str:
    return struct.pack("<BBHHHHH",
                       3,  # sine waveform
                       0,
                       max(start_freq, 1),
                       duration,
                       constrain(start_vol, 0, 1024),
                       constrain(end_vol, 0, 1024),
                       max(end_freq, 1)
                       ).hex()


def moving_average(arr, window_size=3):
    """Simple smoothing with a centered moving average."""
    if window_size < 2:
        return arr
    return np.convolve(arr, np.ones(window_size)/window_size, mode="same")


def audio_to_makecode_arcade(data, sample_rate, period) -> str:
    spectrogram_frequency = period / 1000
    if can_log:
        print(f"Generating spectrogram with a period of {period} ms.")

    f, t, Sxx = scipy.signal.spectrogram(
        data,
        sample_rate,
        nperseg=round(spectrogram_frequency * sample_rate)
    )

    # --- Frequency bucket ranges ---
    frequency_buckets = [50, 159, 200, 252, 317, 400, 504, 635, 800, 1008,
                         1270, 1600, 2016, 2504, 3200, 4032, 5080, 7000, 9000, 10240]

    max_freqs = 30
    loudest_indices = np.argsort(Sxx, axis=0)[-max_freqs:]
    loudest_frequencies = f[loudest_indices].transpose()
    loudest_amplitudes = Sxx[loudest_indices, np.arange(Sxx.shape[1])].transpose()
    max_amp = np.max(Sxx)

    sound_instruction_buffers = [""] * len(frequency_buckets)

    for slice_index in range(len(loudest_frequencies)):
        freqs = loudest_frequencies[slice_index]
        amps = loudest_amplitudes[slice_index]

        # Smooth amplitudes a little
        amps = moving_average(amps, window_size=3)

        for bucket_index in range(len(frequency_buckets)):
            low = frequency_buckets[bucket_index - 1] if bucket_index > 0 else 0
            high = frequency_buckets[bucket_index]
            freq_index = -1
            for i in range(len(freqs) - 1, -1, -1):
                if low <= freqs[i] <= high:
                    freq_index = i
                    break
            if freq_index != -1:
                freq = round(freqs[freq_index])
                amp = round(amps[freq_index] / max_amp * 1024)
                sound_instruction_buffers[bucket_index] += create_sound_instruction(freq, freq, amp, amp, period)
            else:
                # silence
                sound_instruction_buffers[bucket_index] += create_sound_instruction(0, 0, 0, 0, period)

    # Wrap each buffer in hex`` properly
    sound_instruction_buffers = [f"hex`{buf}`" for buf in sound_instruction_buffers]

    # Generate final MakeCode TS with shim + wrapper
    code = (
        "namespace music {\n"
        "    //% shim=music::queuePlayInstructions\n"
        "    export function queuePlayInstructions(timeDelta: number, buf: Buffer) { }\n"
        "}\n\n"
        "const soundInstructions = [\n"
        "    " + ",\n    ".join(sound_instruction_buffers) + "\n"
        "];\n\n"
        "for (const instructions of soundInstructions) {\n"
        "    music.queuePlayInstructions(100, instructions);\n"
        "}\n"
    )
    return code


code = audio_to_makecode_arcade(data, sample_rate, spectrogram_period)
if args.output is not None:
    output_path = args.output.expanduser().resolve()
    if can_log:
        print(f"Writing to {output_path}")
    output_path.write_text(code)
else:
    print(code)
