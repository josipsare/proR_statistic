import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest, shapiro, pearsonr, probplot
import seaborn as sns

dir_path = 'answers'

# Loop through each file in the directory
for filename in os.listdir(dir_path):
    file_path = os.path.join(dir_path, filename)

    # Load JSON data
    with open(file_path, 'r') as file:
        data = json.load(file)


    answer_lengths = []
    word_counts = []
    times = []

    for answer in data:
        answer_lengths.append(len(answer['answer']))
        word_counts.append(len(answer['answer'].split()))
        times.append(answer['time'])
    
    # Calculate statistical parameters
    length_mean = np.mean(answer_lengths)
    length_std = np.std(answer_lengths)

    word_count_mean = np.mean(word_counts)
    word_count_std = np.std(word_counts)

    time_mean = np.mean(times)
    time_std = np.std(times)

    length_ks = kstest(answer_lengths, 'norm')
    word_counts_ks = kstest(word_counts, 'norm')
    time_ks = kstest(times, 'norm')

    length_shapiro = shapiro(answer_lengths)
    word_counts_shapiro = shapiro(word_counts)
    time_shapiro = shapiro(times)

    # Calculate Pearson correlation coefficient between string lengths and numbers
    correlation_coefficient, correlation_p_value = pearsonr(answer_lengths, times)

    # Save results to a file
    os.makedirs(filename[:-5], exist_ok=True)
    output_file_path = f"{filename[:-5]}/results.txt" 
    with open(output_file_path, 'w') as output_file:
        output_file.write(f"{filename[:-5]}\n\n")
        output_file.write(f"-----------------------------------\n\n")

        output_file.write(f"Srednja vrijednost duljine odgovora: {length_mean},\nStandardna devijacija duljine odgovora: {length_std}\n\n")
        output_file.write(f"Srednja vrijednost broja rijeci: {word_count_mean}, \nStandardna devijacija broja rijeci: {word_count_std}\n\n")
        output_file.write(f"Srednja vrijednost trajanja: {time_mean}, \nStandardna devijacija trajanja: {time_std}\n\n")

        output_file.write(f"-----------------------------------\n\n")

        output_file.write("Kolmogorov-Smirnovljev Test p-vrijednosti:\n")
        output_file.write(f"Duljine odgovora: {length_ks.pvalue}\n")
        output_file.write(f"Broja rijeci: {word_counts_ks.pvalue}\n")
        output_file.write(f"Trajanja: {time_ks.pvalue}\n\n")

        output_file.write("Shapiro-Wilk Test p-vrijednosti:\n")
        output_file.write(f"Duljine odgovora: {length_shapiro.pvalue}\n")
        output_file.write(f"Broja rijeci: {word_counts_shapiro.pvalue}\n")
        output_file.write(f"Trajanja: {time_shapiro.pvalue}\n\n")

        output_file.write(f"-----------------------------------\n\n")

        output_file.write(f"Pearsonov koeficjent koleracije za duljinu odgovora i trajanje: {correlation_coefficient}\n")
        output_file.write(f"P-vrijednost koleracije: {correlation_p_value}\n")

    # Create Q-Q plot for numbers and save to file
    plt.figure(figsize=(8, 6))
    probplot(times, plot=plt)
    plt.title(f"Q-Q Plot for Times - {filename[:-5]}")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    qq_plot_path = f"{filename[:-5]}/qq_plotTime.png"
    plt.savefig(qq_plot_path)
    plt.close()

    # Create histogram for numbers and save to file
    plt.figure(figsize=(8, 6))
    sns.histplot(times, bins=20, kde=True, color='skyblue', edgecolor='black')
    plt.title(f"Histogram for Times - {filename[:-5]}")
    plt.xlabel("Time")
    plt.ylabel("Density")
    histogram_path = f"{filename[:-5]}/histogramTime.png"
    plt.savefig(histogram_path)
    plt.close()
