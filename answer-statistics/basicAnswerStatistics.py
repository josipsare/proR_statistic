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
    timePerWord = []

    for answer in data:
        answer_lengths.append(len(answer['answer']))
        wordCount = len(answer['answer'].split())
        word_counts.append(wordCount)
        times.append(answer['time'])
        timePerWord.append(answer['time']/wordCount)
    
    # Calculate statistical parameters
    length_mean = np.mean(answer_lengths)
    length_std = np.std(answer_lengths)

    word_count_mean = np.mean(word_counts)
    word_count_std = np.std(word_counts)

    time_mean = np.mean(times)
    time_std = np.std(times)

    timePerWord_mean = np.mean(timePerWord)
    timePerWord_std = np.std(timePerWord)

    length_ks = kstest(answer_lengths, 'norm')
    word_counts_ks = kstest(word_counts, 'norm')
    time_ks = kstest(times, 'norm')
    timePerWord_ks = kstest(timePerWord, 'norm')

    length_shapiro = shapiro(answer_lengths)
    word_counts_shapiro = shapiro(word_counts)
    time_shapiro = shapiro(times)
    timePerWord_shapiro = shapiro(timePerWord)

    correlation_coefficient, correlation_p_value = pearsonr(answer_lengths, times)

    # Calculate percentiles to categorize data into three groups
    percentiles = np.percentile(answer_lengths, [33.3, 66.6])

    # Categorize data into three groups
    group1_indices = [i for i, length in enumerate(answer_lengths) if length <= percentiles[0]]
    group2_indices = [i for i, length in enumerate(answer_lengths) if percentiles[0] < length <= percentiles[1]]
    group3_indices = [i for i, length in enumerate(answer_lengths) if length > percentiles[1]]

    # Extract data for each group
    group1_lengths = [answer_lengths[i] for i in group1_indices]
    group1_times = [times[i] for i in group1_indices]

    group2_lengths = [answer_lengths[i] for i in group2_indices]
    group2_times = [times[i] for i in group2_indices]

    group3_lengths = [answer_lengths[i] for i in group3_indices]
    group3_times = [times[i] for i in group3_indices]

    # Calculate correlation coefficients and p-values for each group
    corr_coefficient1, p_value1 = pearsonr(group1_lengths, group1_times)
    corr_coefficient2, p_value2 = pearsonr(group2_lengths, group2_times)
    corr_coefficient3, p_value3 = pearsonr(group3_lengths, group3_times)

    # Save results to a file
    os.makedirs(filename[:-5], exist_ok=True)
    output_file_path = f"{filename[:-5]}/results.txt" 
    with open(output_file_path, 'w') as output_file:
        output_file.write(f"{filename[:-5]}\n\n")
        output_file.write(f"-----------------------------------\n\n")

        output_file.write(f"Srednja vrijednost duljine odgovora: {length_mean},\nStandardna devijacija duljine odgovora: {length_std}\n\n")
        output_file.write(f"Srednja vrijednost broja rijeci: {word_count_mean}, \nStandardna devijacija broja rijeci: {word_count_std}\n\n")
        output_file.write(f"Srednja vrijednost trajanja: {time_mean}, \nStandardna devijacija trajanja: {time_std}\n\n")
        output_file.write(f"Srednja vrijednost trajanja po broju rijeci: {timePerWord_mean}, \nStandardna devijacija trajanja po broju rijeci: {timePerWord_std}\n\n")

        output_file.write(f"-----------------------------------\n\n")

        output_file.write("Kolmogorov-Smirnovljev Test p-vrijednosti:\n")
        output_file.write(f"Duljine odgovora: {length_ks.pvalue}\n")
        output_file.write(f"Broja rijeci: {word_counts_ks.pvalue}\n")
        output_file.write(f"Trajanja: {time_ks.pvalue}\n")
        output_file.write(f"Trajanja po broju rijeci: {timePerWord_ks.pvalue}\n\n")

        output_file.write("Shapiro-Wilk Test p-vrijednosti:\n")
        output_file.write(f"Duljine odgovora: {length_shapiro.pvalue}\n")
        output_file.write(f"Broja rijeci: {word_counts_shapiro.pvalue}\n")
        output_file.write(f"Trajanja: {time_shapiro.pvalue}\n")
        output_file.write(f"Trajanja po broju rijeci: {timePerWord_shapiro.pvalue}\n\n")

        output_file.write(f"-----------------------------------\n\n")

        output_file.write(f"Pearsonov koeficjent koleracije za duljinu odgovora i trajanje: {correlation_coefficient}\n")
        output_file.write(f"P-vrijednost koleracije: {correlation_p_value}\n\n")

        output_file.write(f"Pearsonov koeficjent koleracije za duljinu odgovora i trajanje:\n")
        output_file.write(f"Kratki odgovori: {corr_coefficient1}\n")
        output_file.write(f"Srednji odgovori: {corr_coefficient2}\n")
        output_file.write(f"Dugi odgovori: {corr_coefficient3}\n\n")
        
        output_file.write(f"P-vrijednost koleracije: {correlation_p_value}\n")
        output_file.write(f"Kratki odgovori: {p_value1}\n")
        output_file.write(f"Srednji odgovori: {p_value2}\n")
        output_file.write(f"Dugi odgovori: {p_value3}\n")

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

    # Create Q-Q plot for numbers and save to file
    plt.figure(figsize=(8, 6))
    probplot(timePerWord, plot=plt)
    plt.title(f"Q-Q Plot for TimePerWord - {filename[:-5]}")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    qq_plot_path = f"{filename[:-5]}/qq_plotTimePerWord.png"
    plt.savefig(qq_plot_path)
    plt.close()

    # Create histogram for numbers and save to file
    plt.figure(figsize=(8, 6))
    sns.histplot(timePerWord, bins=20, kde=True, color='skyblue', edgecolor='black')
    plt.title(f"Histogram for Times - {filename[:-5]}")
    plt.xlabel("TimePerWord")
    plt.ylabel("Density")
    histogram_path = f"{filename[:-5]}/histogramTimePerWord.png"
    plt.savefig(histogram_path)
    plt.close()
