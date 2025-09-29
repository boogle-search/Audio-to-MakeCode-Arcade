import argparse
import numpy as np
import scipy.io.wavfile
import scipy.signal
from pathlib import Path

def audio_to_makecode_arcade(data, sample_rate, spectrogram_period):
    # Convert stereo to mono if needed
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Normalize data
    data = data.astype(np.float32)
    data /= np.max(np.abs(data))

    # Spectrogram
    nperseg = int(sample_rate * spectrogram_period / 1000)
    noverlap = nperseg // 2
    freqs, times, Sxx = scipy.signal.spectrogram(
        data, fs=sample_rate, nperseg=nperseg, noverlap=noverlap
    )

    # Gather top frequencies per slice
    num_top = 20
    loudest_frequencies = np.argsort(Sxx, axis=0)[-num_top:]
    sound_instruction_buffers = []

    for slice_index in range(Sxx.shape[1]):
        top_indices = loudest_frequencies[:, slice_index]
        freq = round(np.average(freqs[top_indices]))
        amp = np.max(Sxx[top_indices, slice_index])
        if amp > 0.01:  # threshold to skip silence
            buf = f"music.createSoundEffect(WaveShape.Sine, {freq}, {freq}, 255, 0, {spectrogram_period}, SoundExpressionEffect.None, InterpolationCurve.Linear)"
            sound_instruction_buffers.append(buf)

    # TypeScript output
    code = f"""namespace music {{
    //% shim=music::playInstructions
    export function playInstructions(timeDelta: number, buf: Buffer) {{}}
}}

const soundInstructions: Buffer[] = [
    {",\n    ".join(sound_instruction_buffers)}
];

for (const instructions of soundInstructions) {{
    music.playInstructions(100, instructions);
}}
"""
    return code

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--period", type=int, default=25)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    spectrogram_period = args.period

    print(f"Opening audio {input_path}")
    sample_rate, data = scipy.io.wavfile.read(input_path)
    print(f"Audio has {len(data)} samples at {sample_rate} Hz")

    code = audio_to_makecode_arcade(data, sample_rate, spectrogram_period)

    with open(output_path, "w") as f:
        f.write(code)

    print(f"Wrote MakeCode file to {output_path}")

if __name__ == "__main__":
    main()
