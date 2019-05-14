# build_dictionary, build_features and build_labels functions are inspired from below github repo but modified
# alameenkhader/spam_classifier

import os
import numpy as np
import re
import math
import operator

knn_k = 3
#np.seterr(divide='print', invalid='print', over='print', under='print')

learning_legitimate_class_word_size = 0
learning_legitimate_class_corpus_size = 0
learning_spam_class_word_size = 0
learning_spam_class_corpus_size = 0


def compute_cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norma = np.linalg.norm(vec1)
    normb = np.linalg.norm(vec2)
    cos = dot_product / (norma * normb)
    return cos


# Creates dictionary from all the emails in the directory
def build_dictionary(dir, val):
    global learning_legitimate_class_word_size
    global learning_legitimate_class_corpus_size
    global learning_spam_class_word_size
    global learning_spam_class_corpus_size

    # Read the file names
    emails = os.listdir(dir)
    emails.sort()
    # Array to hold all the words in the emails
    dictionary = []

    # Collecting all words from those emails
    for email in emails:
        m = open(os.path.join(dir, email), encoding='latin-1')
        for i, line in enumerate(m):
            if i == 2:  # Body of email is only 3rd line of text file
                words = line.split()
                dictionary += words

    # Removes punctuations and non alphabets

    for i in range(8):
        for index, word in enumerate(dictionary):
            if (not word.isalpha()) or (len(word) == 1):
                del dictionary[index]

    if val == 0:
        learning_legitimate_class_word_size = len(dictionary)
    elif val == 1:
        learning_spam_class_word_size = len(dictionary)

    # We now have the array of words, which may have duplicate entries
    dictionary = list(set(dictionary))  # Removes duplicates

    if val == 0:
        learning_legitimate_class_corpus_size = len(dictionary)
    elif val == 1:
        learning_spam_class_corpus_size = len(dictionary)

    return dictionary


def build_features(dir, dictionary):
    # Read the file names
    emails = os.listdir(dir)
    emails.sort()
    # ndarray to have the features
    features_matrix = np.zeros((len(emails), len(dictionary)))

    for email_index, email in enumerate(emails):
        m = open(os.path.join(dir, email), encoding='latin-1')
        for line_index, line in enumerate(m):
            if line_index == 2:
                words = line.split()
                for word_index, word in enumerate(dictionary):
                    if word in words:
                        features_matrix[email_index, word_index] = 1
                    else:
                        features_matrix[email_index, word_index] = 0
    return features_matrix


def build_labels(dir):
    # Read the file names
    emails = os.listdir(dir)
    emails.sort()
    # ndarray of labels
    labels_matrix = np.zeros(len(emails))

    for index, email in enumerate(emails):
        labels_matrix[index] = 1 if re.search('spms*', email) else 0

    return labels_matrix


def build_tfidf(dir, dictionary, doc_freq):
    # Read the file names
    emails = os.listdir(dir)
    emails.sort()
    # ndarray to have the features
    tf_idf_matrix = np.zeros((len(emails), len(dictionary)))

    for email_index, email in enumerate(emails):
        m = open(os.path.join(dir, email), encoding='latin-1')
        for line_index, line in enumerate(m):
            if line_index == 2:
                words = line.split()
                for word_index, word in enumerate(dictionary):
                    if word in words:
                        tf_idf_matrix[email_index, word_index] = (1 + math.log10(words.count(word))) * math.log10(len(emails) / doc_freq[word_index])

    return tf_idf_matrix


train_dir_legitimate = './dataset/training/legitimate'
dictionary_legitimate = build_dictionary(train_dir_legitimate, 0)

train_dir_spam = './dataset/training/spam'
dictionary_spam = build_dictionary(train_dir_spam, 1)

global_learning_dictionary = list(set(dictionary_legitimate + dictionary_spam))

features_train_legitimate = build_features(train_dir_legitimate, global_learning_dictionary)

features_train_spam = build_features(train_dir_spam, global_learning_dictionary)

t1, t2 = features_train_legitimate.shape

a_legitimate = []
for i in range(len(global_learning_dictionary)):
    a_legitimate.append(0)

for i in range(t1):
    a_legitimate += features_train_legitimate[i]


labels_train_legitimate = build_labels(train_dir_legitimate)

t1, t2 = features_train_spam.shape

a_spam = []
for i in range(len(global_learning_dictionary)):
    a_spam.append(0)

for i in range(t1):
    a_spam += features_train_spam[i]

a_combined = []
for i in range(len(global_learning_dictionary)):
    a_combined.append(a_spam[i]+a_legitimate[i])

labels_train_spam = build_labels(train_dir_spam)

train_legitimate_tfidf = build_tfidf(train_dir_legitimate, global_learning_dictionary, a_combined)

train_spam_tfidf = build_tfidf(train_dir_spam, global_learning_dictionary, a_combined)

t1, t2 = train_legitimate_tfidf.shape

a_legitimate_tfidf = []
for i in range(len(global_learning_dictionary)):
    a_legitimate_tfidf.append(0)

for i in range(t1):
    a_legitimate_tfidf += train_legitimate_tfidf[i]

result_args = np.argpartition(a_legitimate_tfidf, -20)[-20:]

legitimate_tfidf_weights = a_legitimate_tfidf[result_args]

top_legitimate_tfids_features = []

for i in range(len(result_args)):
    top_legitimate_tfids_features.append(global_learning_dictionary[result_args[i]])

print("Top 20 words with the highest total TF-IDF scores in the non-spam training set:")
print(top_legitimate_tfids_features)

with open('Top_20_legitimate_features.txt', 'w+') as the_file:
    for i in range(len(top_legitimate_tfids_features)):
        the_file.write('{} - {}\n'.format((i + 1), top_legitimate_tfids_features[i]))

t1, t2 = train_spam_tfidf.shape

a_spam_tfidf = []
for i in range(len(global_learning_dictionary)):
    a_spam_tfidf.append(0)

for i in range(t1):
    a_spam_tfidf += train_spam_tfidf[i]

result_args = np.argpartition(a_spam_tfidf, -20)[-20:]

spam_tfidf_weights = a_spam_tfidf[result_args]

top_spam_tfids_features = []

for i in range(len(result_args)):
    top_spam_tfids_features.append(global_learning_dictionary[result_args[i]])

print("Top 20 words with the highest total TF-IDF scores in the spam training set:")
print(top_spam_tfids_features)

with open('Top_20_spam_features.txt', 'w+') as the_file:
    for i in range(len(top_spam_tfids_features)):
        the_file.write('{} - {}\n'.format((i + 1), top_spam_tfids_features[i]))

t1, t2 = train_legitimate_tfidf.shape
centroid_legitimate = a_legitimate_tfidf / t1

t1, t2 = train_spam_tfidf.shape
centroid_spam = a_spam_tfidf / t1

test_dir_legitimate = './dataset/test/legitimate'

test_legitimate_tfidf = build_tfidf(test_dir_legitimate, global_learning_dictionary, a_combined)

test_dir_spam = './dataset/test/spam'

test_spam_tfidf = build_tfidf(test_dir_spam, global_learning_dictionary, a_combined)

t1, t2 = test_legitimate_tfidf.shape

n1, n2 = train_legitimate_tfidf.shape

a = []
b = []

labels_test_legitimate = []

for i in range(t1):
    x_leg = compute_cosine_similarity(test_legitimate_tfidf[i], centroid_legitimate)
    x_spam = compute_cosine_similarity(test_legitimate_tfidf[i], centroid_spam)
    if x_leg > x_spam:
        labels_test_legitimate.append(0)
    else:
        labels_test_legitimate.append(1)

labels_test_spam = []

for i in range(t1):
    x_leg = compute_cosine_similarity(test_spam_tfidf[i], centroid_legitimate)
    x_spam = compute_cosine_similarity(test_spam_tfidf[i], centroid_spam)
    if x_leg > x_spam:
        labels_test_spam.append(0)
    else:
        labels_test_spam.append(1)


TP1 = labels_test_legitimate.count(0)
FP1 = len(labels_train_legitimate) - labels_test_legitimate.count(0)
FN1 = len(labels_train_spam) - labels_test_spam.count(1)
TN1 = labels_test_spam.count(1)



TP2 = labels_test_spam.count(1)
FP2 = len(labels_train_spam) - labels_test_spam.count(1)
FN2 = len(labels_train_legitimate) - labels_test_legitimate.count(0)
TN2 = labels_test_legitimate.count(0)



Precision_1 = TP1 / (TP1 + FP1)
Recall_1 = TP1 / (TP1 + FN1)
F_measure_1 = 2 * Precision_1 * Recall_1 / (Precision_1 + Recall_1)

Precision_2 = TP2 / (TP2 + FP2)
Recall_2 = TP2 / (TP2 + FN2)
F_measure_2 = 2 * Precision_2 * Recall_2 / (Precision_2 + Recall_2)

macro_averaged_precision = (Precision_1 + Precision_2) / 2
micro_averaged_precision = (TP1 + TP2) / (TP1 + FP1 + TP2 + FP2)

macro_averaged_recall = (Recall_1 + Recall_2) / 2
micro_averaged_recall = (TP1 + TP2) / (TP1 + FN1 + TP2 + FN2)

print("\n------------------\n")

print("Dataset (learning) legitimate class total word size is: {}".format(learning_legitimate_class_word_size))

print("Dataset (learning) legitimate class corpus size is: {}".format(learning_legitimate_class_corpus_size))

print("-------------------------------------------------")

print("\nDataset (learning) spam class total word size is: {}".format(learning_spam_class_word_size))

print("Dataset (learning) spam class corpus size is: {}".format(learning_spam_class_corpus_size))

print("\n-------------------------------------------------")

print("\nLearning dataset all classes corpus size is: {}\n".format(len(global_learning_dictionary)))

print("\n-------------------------------------------------\n")


print("Precision for legitimate mail test set is: %{:.4f} percent.".format(Precision_1 * 100))
print("Recall for legitimate mail test set is: %{:.4f} percent.".format(Recall_1 * 100))
print("F_Measure for legitimate mail test set is: %{:.4f} percent.".format(F_measure_1 * 100))

print("Precision for spam mail test set is: %{:.4f} percent.".format(Precision_2 * 100))
print("Recall for spam mail test set is: %{:.4f} percent.".format(Recall_2 * 100))
print("F_Measure for spam mail test set is: %{:.4f} percent.".format(F_measure_2 * 100))

print("Macro Averaged Precision is: %{:.4f} percent.".format(macro_averaged_precision * 100))
print("Micro Averaged Precision is: %{:.4f} percent.".format(micro_averaged_precision * 100))

print("Macro Averaged Recall is: %{:.4f} percent.".format(macro_averaged_recall * 100))
print("Micro Averaged Recall is: %{:.4f} percent.".format(micro_averaged_recall * 100))
