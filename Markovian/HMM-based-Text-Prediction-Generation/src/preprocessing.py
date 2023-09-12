from nltk.tokenize import word_tokenize
import pickle
import operator
import string

def process_data(filename):
    """
    Tokenize the sonnets
    :param filename: file path
    :return: a dictionary of all words, a list of all lines, and a list of all poems
    """
    all_lines = []  # a list of all lines
    all_words = {}  # a dictionary of all words with wordcount
    all_poems = []  # a list of all poems

    punctuations = list(string.punctuation)  # all punctuations

    poem_temp = []
    with open(filename) as f:
        for line in f.readlines():
            line_tokens = [word.lower() for word in word_tokenize(line)]
            line_tokens = [x for x in line_tokens if x not in punctuations]

            if len(line_tokens) > 1:
                for word in line_tokens:
                    if word in all_words:
                        all_words[word] += 1
                    else:
                        all_words[word] = 1
                all_lines.append(line_tokens)
                poem_temp += line_tokens
            elif len(line_tokens) == 1:
                all_poems.append(poem_temp)
                poem_temp = []

    all_poems.append(poem_temp)
    all_poems = all_poems[1:]  # remove [] in the first index

    return all_words, all_poems, all_lines


def compute_bigram_count(all_lines, all_words):
    """
    Count all bigrams in all_lines

    :param all_lines: a list of list of word token
    :param all_words: a dictionary of words with wordcount
    :return: a list of bigrams with the counts
    """
    bigram_list = {}
    for line in all_lines:
        for ind in range(len(line)-1):
            if (line[ind], line[ind+1]) in bigram_list:
                bigram_list[(line[ind], line[ind+1])] += 1
            else:
                bigram_list[(line[ind], line[ind + 1])] = 1

    bigram_list = sorted(bigram_list.items(), key=operator.itemgetter(1), reverse=True)

    for ind, (bigram, value) in enumerate(bigram_list):
        bigram_list[ind] = (bigram, all_words[bigram[0]], all_words[bigram[1]], value)

    return bigram_list


def replace_bigram(all_bigrams, all_lines, threshold=20):
    """
    Replace pair of word tokens with bigrams if the frequency
    of appearance is above threshold

    :param all_bigrams: a list of bigrams
    :param all_lines: a list of list of word token
    :param threshold: cut off for bigram's appearance frequency
    :return: a list of list of word token and a list of all words including the bigrams
    """
    bigrams_that_matters = [item[0] for item in all_bigrams if item[-1] >= threshold]

    all_words = set()
    new_lines = []
    for line in all_lines:
        for ind in range(len(line) - 1):
            try:
                if (line[ind], line[ind + 1]) in bigrams_that_matters:
                    line[ind] = line[ind] + ' ' + line[ind + 1]
                    del line[ind+1]
            except IndexError:
                pass

        line = [item for item in line if item not in ["'s", "'t"]]
        new_lines.append(line)
        all_words.update(line)

    return new_lines, list(all_words)


def compute_rhythm_dictionary(all_lines):
    """
    Create the rhythm dictionary

    :param all_lines: a list of all lines in the sonnet
    :return: a set of tuple with pair of words that rhyme
    """
    all_rhythm = set()
    poem = 0
    begin = 0
    stanza = 0
    tot_line = 0
    rhythm1 = []
    rhythm2 = []

    punctuations = list(string.punctuation)

    for line in all_lines:

        line = [item for item in line if item not in punctuations]

        if poem == 98:  # special sonnet
            if stanza == 0:
                rhythm1.append(line[-1])
                tot_line += 1

                if tot_line == 5:
                    all_rhythm.update([tuple(sorted([rhythm1[0], rhythm1[2]])),
                                      tuple(sorted([rhythm1[1], rhythm1[3]])),
                                      tuple(sorted([rhythm1[2], rhythm1[4]]))])
                    rhythm1 = []
                    stanza += 1
                    tot_line = 0

            elif stanza < 3:
                if begin == 0:
                    rhythm1.append(line[-1])
                    begin = 1
                elif begin == 1:
                    rhythm2.append(line[-1])
                    begin = 0
                tot_line += 1

                if tot_line == 4:
                    all_rhythm.update([tuple(sorted(rhythm1)), tuple(sorted(rhythm2))])
                    tot_line = 0
                    stanza += 1
                    rhythm1 = []
                    rhythm2 = []

            else:
                rhythm1.append(line[-1])
                tot_line += 1

                if tot_line == 2:
                    tot_line = 0
                    all_rhythm.add(tuple(sorted(rhythm1)))
                    stanza = 0
                    rhythm1 = []
                    poem += 1

        elif poem == 125:  # special sonnet
            rhythm1.append(line[-1])
            tot_line += 1

            if tot_line == 2:
                tot_line = 0
                all_rhythm.add(tuple(sorted(rhythm1)))
                stanza += 1
                rhythm1 = []

            if stanza == 6:
                poem += 1
                stanza = 0

        else:  # all other sonnets
            if stanza < 3:
                if begin == 0:
                    rhythm1.append(line[-1])
                    begin = 1
                elif begin == 1:
                    rhythm2.append(line[-1])
                    begin = 0
                tot_line += 1

                if tot_line == 4:
                    all_rhythm.update([tuple(sorted(rhythm1)), tuple(sorted(rhythm2))])
                    tot_line = 0
                    stanza += 1
                    rhythm1 = []
                    rhythm2 = []

            else:
                rhythm1.append(line[-1])
                tot_line += 1

                if tot_line == 2:
                    tot_line = 0
                    all_rhythm.add(tuple(sorted(rhythm1)))
                    stanza = 0
                    rhythm1 = []
                    poem += 1

    return all_rhythm


def convert_to_integer(all_words, all_lines):
    """
    Convert from words to integer

    :param all_words: a list of all words
    :param all_lines: a list of list of word token
    :return: a list of list of word token in integer
    and the corresponding dictionary that maps the integer to words
    """
    dictionary = {i: word for i, word in enumerate(all_words)}
    new_lines = []
    for line in all_lines:
        new_line = []
        for word in line:
            if word in all_words:####all_words.index(word)>=0: usman
                new_line.append(all_words.index(word))
        if new_line:
            new_lines.append(new_line)

    return new_lines, dictionary


def save_obj(obj, name):
    """
    Save object
    """
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

#########################################################################
#                               Main code                               #
#########################################################################
all_words, all_poems, all_lines = process_data('../dataset/shakespeare.txt')

all_bigrams = compute_bigram_count(all_lines, all_words)

all_rhythm = compute_rhythm_dictionary(all_lines)

training_line, training_symbols = replace_bigram(all_bigrams, all_lines, threshold=15)
training_poem, training_symbols = replace_bigram(all_bigrams, all_poems, threshold=15)

training_line_int, dictionary = convert_to_integer(training_symbols, training_line)
training_poem_int, dictionary = convert_to_integer(training_symbols, training_poem)

# Reverse word dictionary where for each word, return the corresponding integer
dictionary_2 = {v: k for k, v in dictionary.items()}

# Reverse the sequence
training_line_int.reverse()
training_poem_int.reverse()

# Save all data
save_obj(training_line_int, './pp_data/training_data')
save_obj(training_poem_int, './pp_data/training_poem_data')
save_obj(dictionary, './pp_data/word_dictionary')
save_obj(dictionary_2, './pp_data/word_dictionary_reverse')
save_obj(training_symbols, './pp_data/symbols')
save_obj(all_rhythm, './pp_data/rhythm')

