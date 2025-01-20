import numpy as np
import soundfile as sf
import pyworld as pw
import os
import glob

frame_period = 5.0
semitones_shift_list = [-12, -6, 6, 12]
input_folder = "test_features"
output_folder = "test_shifted"

def read_f0_file(f0_file_path):
    f0_times = []
    f0_values = []
    with open(f0_file_path, "r") as f:
        for line in f:
            time_str, f0_str = line.strip().split('\t')
            time = float(time_str)
            f0 = float(f0_str)
            f0_values.append(f0)
            f0_times.append(time)
    return np.array(f0_times), np.array(f0_values)

def interpolate_f0(f0_times, f0_values, target_times):
    f0_interpolated = np.interp(target_times, f0_times, f0_values)
    f0_interpolated[np.isnan(f0_interpolated) | (f0_interpolated <= 0)] = 0.0
    return f0_interpolated

def pitch_shift(harmonic_waveform, sr, f0_interpolated, shift_factor, frame_period):
    modified_f0 = f0_interpolated * shift_factor

    frame_period_sec = frame_period / 1000.0
    time_axis = np.arange(len(f0_interpolated)) * frame_period_sec

    fft_size = pw.get_cheaptrick_fft_size(sr)
    spectrogram = pw.cheaptrick(
        harmonic_waveform, f0_interpolated, time_axis, sr, fft_size=fft_size
    )

    number_of_bins = fft_size // 2 + 1
    aperiodicity = np.ones((len(modified_f0), number_of_bins))
    aperiodicity[modified_f0 > 0, :] = 0.0

    synthesized_waveform = pw.synthesize(
        modified_f0, spectrogram, aperiodicity, sr, frame_period=frame_period
    )
    return synthesized_waveform

def apply_fade(audio, sr, fade_duration=0.01):
    fade_samples = int(sr * fade_duration)
    if fade_samples > 0 and fade_samples*2 < len(audio):
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        audio[:fade_samples] *= fade_in
        audio[-fade_samples:] *= fade_out
    return audio

def process_files(harmonic_path, noise_path, f0_path, output_path_base, semitone_shift):
    shift_factor = 2 ** (semitone_shift / 12.0)

    harmonic_waveform, sr = sf.read(harmonic_path)
    if harmonic_waveform.ndim > 1:
        harmonic_waveform = harmonic_waveform.mean(axis=1)
    if harmonic_waveform.dtype.kind in ["i", "u"]:
        harmonic_waveform = harmonic_waveform.astype(np.float64) / np.iinfo(harmonic_waveform.dtype).max

    f0_times, f0_values = read_f0_file(f0_path)
    frame_period_sec = frame_period / 1000.0
    total_time = len(harmonic_waveform) / sr
    number_of_frames = int(total_time / frame_period_sec) + 1
    target_times = np.arange(number_of_frames) * frame_period_sec
    f0_interpolated = interpolate_f0(f0_times, f0_values, target_times)

    synthesized_waveform = pitch_shift(harmonic_waveform, sr, f0_interpolated, shift_factor, frame_period)

    noise_waveform, sr_noise = sf.read(noise_path)
    if noise_waveform.ndim > 1:
        noise_waveform = noise_waveform.mean(axis=1)
    if noise_waveform.dtype.kind in ["i", "u"]:
        noise_waveform = noise_waveform.astype(np.float64) / np.iinfo(noise_waveform.dtype).max

    min_length = min(len(synthesized_waveform), len(noise_waveform))
    synthesized_waveform = synthesized_waveform[:min_length]
    noise_waveform = noise_waveform[:min_length]

    combined_waveform = synthesized_waveform + noise_waveform
    combined_waveform = apply_fade(combined_waveform, sr, fade_duration=0.01)

    shift_sign = "+" if semitone_shift > 0 else "-"
    output_path = f"{output_path_base}{shift_sign}{abs(semitone_shift)}.wav"

    sf.write(output_path, combined_waveform, sr)
    print(f"Pitch-shifted audio (shift {semitone_shift} semitones) saved to {output_path}")

def main():
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    harmonic_files = glob.glob(os.path.join(input_folder, "*_harmonic.wav"))

    for harmonic_path in harmonic_files:
        base_name = os.path.basename(harmonic_path).replace("_harmonic.wav", "")

        noise_path = os.path.join(input_folder, f"{base_name}_noise.wav")
        f0_path = os.path.join(input_folder, f"{base_name}_f0.txt")

        if not os.path.exists(noise_path):
            print(f"Noise file not found for {base_name}, skipping.")
            continue
        if not os.path.exists(f0_path):
            print(f"F0 file not found for {base_name}, skipping.")
            continue

        output_path_base = os.path.join(output_folder, base_name)

        print(f"Processing {base_name}...")
        for semitone_shift in semitones_shift_list:
            try:
                process_files(harmonic_path, noise_path, f0_path, output_path_base, semitone_shift)
            except Exception as e:
                print(f"Error processing {base_name} with shift {semitone_shift}: {e}")

if __name__ == "__main__":
    main()
