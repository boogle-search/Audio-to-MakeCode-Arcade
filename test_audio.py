import struct
from pathlib import Path
import numpy as np
import scipy

# --- CONFIG ---
input_path = Path("audio/test.wav")  # put your test WAV here
output_path = Path("output/test_output.ts")
period = 25  # ms

# --- FUNCTIONS ---
def constrain(value, min_value, max_value):
    return min(max(value, min_value), max_value)

def create_sound_instruction(start_freq, end_freq, start_vol, end_vol, duration):
    return struct.pack("<BBHHHHH",
                       3, 0, max(start_freq, 1), duration,
                       constrain(start_vol, 0, 1024),
                       constrain(end_vol, 0, 1024),
                       max(end_freq, 1)
                       ).hex()

def audio_to_makecode(data, sample_rate, period):
    spectrogram_frequency = period / 1000
    f, t, Sxx = scipy.signal.spectrogram(
        data,
        sample_rate,
        nperseg=round(spectrogram_frequency * sample_rate),
        noverlap=round(spectrogram_frequency * sample_rate // 2)
    )

    frequency_buckets = [50, 159, 200, 252, 317, 400, 504, 635, 800, 1008,
                         1270, 1600, 2016, 2504, 3200, 4032, 5080, 7000, 9000, 10240]

    max_freqs = 16
    loudest_indices = np.argsort(Sxx, axis=0)[-max_freqs:]
    loudest_frequencies = f[loudest_indices].transpose()
    loudest_amplitudes = Sxx[loudest_indices, np.arange(Sxx.shape[1])].transpose()
    max_amp = np.max(Sxx)

    sound_instruction_buffers = [""] * len(frequency_buckets)

    for slice_index in range(len(loudest_frequencies)):
        for bucket_index in range(len(frequency_buckets)):
            freqs = loudest_frequencies[slice_index]
            low = frequency_buckets[bucket_index - 1] if bucket_index > 0 else 0
            high = frequency_buckets[bucket_index]
            freq_index = -1
            for i in range(len(freqs)-1, -1, -1):
                if low <= freqs[i] <= high:
                    freq_index = i
                    break
            if freq_index != -1:
                freq = round(freqs[freq_index])
                amp = max(round(loudest_amplitudes[slice_index, freq_index] / max_amp * 1024), 64)
                sound_instruction_buffers[bucket_index] += create_sound_instruction(freq, freq, amp, amp, period)
            else:
                sound_instruction_buffers[bucket_index] += create_sound_instruction(0,0,0,0,period)

    sound_instruction_buffers = [f"hex`{buf}`" for buf in sound_instruction_buffers]

    return (
        "namespace music {\n"
        "    //% shim=music::queuePlayInstructions\n"
        "    export function queuePlayInstructions(timeDelta: number, buf: Buffer) { }\n\n"
        "    export function playInstructions(timeDelta: number, buf: Buffer) {\n"
        "        queuePlayInstructions(timeDelta, buf);\n"
        "    }\n"
        "}\n\n"
        f"const soundInstructions = [\n    {',\n    '.join(sound_instruction_buffers)}\n];\n\n"
        "for (const instructions of soundInstructions) {\n"
        "    music.playInstructions(100, instructions);\n"
        "}\n"
    )

# --- MAIN ---
sample_rate, data = scipy.io.wavfile.read(input_path)
if len(data.shape) > 1 and data.shape[1] > 1:
    data = data[:, 0]

output_path.write_text(audio_to_makecode(data, sample_rate, period))
print(f"Test MakeCode TS generated at {output_path}")
