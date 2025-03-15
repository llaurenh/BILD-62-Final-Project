import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_average_normal_ecg(file_path):
    data = np.loadtxt(file_path, delimiter=",", skiprows=1)
    normal_ecgs = data[data[:, -1] == 0, :-1]
    return np.mean(normal_ecgs, axis=0) if normal_ecgs.size > 0 else None

def plot_ecg(file_path, index):
    data = np.loadtxt(file_path, delimiter=",", skiprows=1)
    if index < 0 or index >= data.shape[0]:
        raise ValueError("Invalid index. Please provide a valid patient index.")
    ecg_signal = data[index, :-1]
    plt.figure(figsize=(10, 4))
    plt.plot(ecg_signal, label=f"ECG Signal (Patient {index})", color="blue")
    plt.title(f"ECG Plot - Patient {index}")
    plt.xlabel("Time (ms)")
    plt.ylabel("ECG Amplitude")
    plt.legend()
    plt.grid()
    plt.show()

def plot_ecg_with_overlay(file_path, index):
    data = np.loadtxt(file_path, delimiter=",", skiprows=1)
    if index < 0 or index >= data.shape[0]:
        raise ValueError("Invalid index. Please provide a valid patient index.")
    ecg_signal = data[index, :-1]
    label = int(data[index, -1])
    std_dev = np.std(ecg_signal)
    max_value = np.max(ecg_signal)
    min_value = np.min(ecg_signal)
    is_abnormal = label == 1
    abnormal_reason = ""
    if is_abnormal:
        if std_dev > 0.5:
            abnormal_reason += "High variance detected. "
        if max_value > 2 or min_value < -2:
            abnormal_reason += "Unusual peaks detected. "
    high_peaks = np.where(ecg_signal > 2)[0]
    low_peaks = np.where(ecg_signal < -2)[0]
    window_size = 10
    rolling_std = np.array([np.std(ecg_signal[i:i+window_size]) for i in range(len(ecg_signal)-window_size+1)])
    high_variance_regions = np.where(rolling_std > 0.5)[0]
    average_normal_ecg = get_average_normal_ecg(file_path)
    plt.figure(figsize=(10, 4))
    plt.plot(ecg_signal, label="Abnormal ECG" if is_abnormal else "ECG Signal", color="blue")
    if is_abnormal and average_normal_ecg is not None:
        plt.plot(average_normal_ecg, label="Normal ECG (Reference)", color="green", linestyle="dashed")
    plt.scatter(high_peaks, ecg_signal[high_peaks], color="red", label="High Peaks", zorder=3)
    plt.scatter(low_peaks, ecg_signal[low_peaks], color="purple", label="Low Peaks", zorder=3)
    for idx in high_variance_regions:
        plt.axvspan(idx, idx+window_size, color='yellow', alpha=0.3, label="High Variance Region" if idx == high_variance_regions[0] else "", zorder=2)
    plt.title(f"ECG Plot KEY - Patient {index}")
    plt.xlabel("Time (ms)")
    plt.ylabel("ECG Amplitude")
    plt.legend()
    plt.grid()
    if is_abnormal:
        plt.figtext(0.5, 0.01, f"Abnormality: {abnormal_reason}", wrap=True, horizontalalignment='center', fontsize=10, color="red")
    plt.show()

def ecg_quiz(file_path, ecg_file):
    df = pd.read_csv(file_path)
    score = 0
    patient_ids = df['Patient'].unique()
    for patient_id in patient_ids:
        print(f"\nLoading ECG for Patient {patient_id}...")
        plot_ecg(ecg_file, patient_id)
        patient_questions = df[df['Patient'] == patient_id]
        for index, row in patient_questions.iterrows():
            question = row['Question']
            correct_answer = str(row['Answer']).strip().lower()
            print(f"Patient {patient_id}: {question}")
            user_answer = input("Your answer: ").strip().lower()
            if user_answer == correct_answer:
                print("Correct!\n")
                score += 1
            else:
                print(f"Wrong. The correct answer is: {correct_answer}\n")
        print(f"\nShowing annotated ECG for Patient {patient_id}...")
        plot_ecg_with_overlay(ecg_file, patient_id)
    print(f"Final score: {score}/{len(df)}")
    return score

def total_score(file_paths, ecg_file):
    total_correct = 0
    total_questions = 0
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        total_questions += len(df)
        total_correct += ecg_quiz(file_path, ecg_file)
    print(f"Total score across all patients: {total_correct}/{total_questions}")
    return total_correct, total_questions
