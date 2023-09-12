import sys
import random
import pickle
from nltk.corpus import cmudict
import string
from countsyl import count_syllables


def syllables_in_word(word, phoneme_dict):
    """
    Count the number of syllables in the string argument 'word'
    :param word: string
    :return: number of syllables
    """
    if word in phoneme_dict:
        return len([ph for ph in phoneme_dict[word] if ph.strip(string.ascii_letters)])#usman
    else:
        return 0


def load_obj(name):
    """
    Load data
    """
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

# Load in observation matrix and word dictionary (int,word pairs)
O = load_obj('./data/observation_matrix_line_10')# all changed from 20 to 10
word_dict_rev = load_obj('./pp_data/word_dictionary_reverse')
word_dict = load_obj('./pp_data/word_dictionary')
A = load_obj('./data/transition_matrix_line_10')
rhythm_dict = load_obj('./pp_data/rhythm')#changed by usman
HMM = load_obj('./data/HMM_line_10')

##print(len(sys.argv))

if len(sys.argv)<3:
    print("Missing some arguments... It should be python3 prediction.py <text file name> <number of next words to be predicted (between 1 to 5 inclusive, For all other values=1)>")

predsRequired=int(sys.argv[2])

if predsRequired<1 or predsRequired>5:
    predsRequired = 1

f = open(sys.argv[1], "r")
line=f.readline()
seqOfWords=line.split()
##print(len(seqOfWords))
f.close()

if len(seqOfWords)==0:
    print("Text File is empty... please fil that first.")
    ##return 0

L = len(O)
D = len(O[0])

phoneme_dict = dict(cmudict.entries())

rhyme_pair = [0. for _ in range(7)]
rand_selector = [0. for _ in range(7)]
##print(word_dict_rev)
for i in range(7):#orignially 7 changed by usman
    rhyme_pair[i] = random.sample(rhythm_dict, 1)[0]
    rand_selector[i] = random.randint(0, 1)

for l in range(1):#original usman 14
    # Set first word
    if l == 0 or l == 1:
        first_word5 = rhyme_pair[l][rand_selector[l]]
        first_index5 = word_dict_rev[first_word5]
    elif l == 2 or l == 3:
        if rand_selector[l-2] == 1:
            first_word5 = rhyme_pair[l-2][0]
            first_index5 = word_dict_rev[first_word5]
        else:
            first_word5 = rhyme_pair[l-2][1]
            first_index5 = word_dict_rev[first_word5]
    elif l == 4 or l == 5:
        first_word5 = rhyme_pair[l-2][rand_selector[l-2]]
        first_index5 = word_dict_rev[first_word5]
    elif l == 6 or l == 7:
        if rand_selector[l-4] == 1:
            first_word5 = rhyme_pair[l-4][0]
            first_index5 = word_dict_rev[first_word5]
        else:
            first_word5 = rhyme_pair[l-4][1]
            first_index5 = word_dict_rev[first_word5]
    elif l == 8 or l == 9:
        first_word5 = rhyme_pair[l-4][rand_selector[l-4]]
        first_index5 = word_dict_rev[first_word5]
    elif l == 10 or l == 11:
        if rand_selector[l-6] == 1:
            first_word5 = rhyme_pair[l-6][0]
            first_index5 = word_dict_rev[first_word5]
        else:
            first_word5 = rhyme_pair[l-6][1]
            first_index5 = word_dict_rev[first_word5]
    elif l == 12:
        first_word5 = rhyme_pair[l-6][rand_selector[l-6]]
        first_index5 = word_dict_rev[first_word5]
    elif l == 13:
        if rand_selector[l-7] == 1:
            first_word5 = rhyme_pair[l-7][0]
            first_index5 = word_dict_rev[first_word5]
        else:
            first_word5 = rhyme_pair[l-7][1]
            first_index5 = word_dict_rev[first_word5]

    first_word=""
    first_index=-1

    for h in range(len(seqOfWords)-1,-1,-1):
        if seqOfWords[h] in word_dict_rev:
            first_word = seqOfWords[h]
            first_index = word_dict_rev[first_word]
            break
    toBeAdded=""
    if first_index==-1:
        toBeAdded = " " + first_word5
        first_word = first_word5
        first_index = first_index5
    ##print(first_word + str(first_index))
    # Get P(y|x) given first x
    P_yx = [0. for _ in range(L)]
    alphas = HMM.forward([first_index], normalize=True)
    betas = HMM.backward([first_index], normalize=True)
    for curr in range(L):
        P_yx[curr] = alphas[1][curr]*betas[0][curr]

    norm = sum(P_yx)
    for curr in range(len(P_yx)):
        P_yx[curr] /= norm

    state = []

    # Initialize first state based on first word
    P_S = P_yx
    r_S = random.uniform(0,1)
    for k in range(L):
        if r_S < sum(P_S[0:k+1]):
            state.append(k)
            break

    prev_k = -1
            
    # Update each state and emission
    i = 0
    syllable_count = count_syllables(first_word)
    if syllable_count == 0:
        syllable_count = syllables_in_word(first_word, phoneme_dict)
    rand_sequence = [first_word]

    while syllable_count < 10:
        # Update each state based on previous state
        P_A = A[state[i]]
        r_A = random.uniform(0,1)
        for k in range(L):
            if r_A < sum(P_A[0:k+1]):
                state.append(k)
                break

        # Determine current emission based on current state
        P_O = O[state[i]]
        r_O = random.uniform(0,1)
        for k in range(D):
            if r_O < sum(P_O[0:k+1]):
                if k == prev_k:
                    state.pop()
                    break
                else:
                    new_word = word_dict[k]
                
                    curr_syllable = count_syllables(new_word)
                    if curr_syllable == 0:
                        curr_syllable = syllables_in_word(new_word, phoneme_dict)
                    syllable_count += curr_syllable
                
                    if syllable_count > 10:
                        state.pop()
                        syllable_count -= curr_syllable
                        break
                    else:
                        rand_sequence.append(new_word)
                        prev_k = k
                        break

    ##rand_sequence.reverse() commented by usman

    # print the sentence
    print("\n Given sequence of words \""+" ".join(seqOfWords)+"\", Complete sequence after adding  next "+str(predsRequired)+" word(s) will be :")
    nextWords9=""
    for x in range(1,predsRequired+1):
        nextWords9 = nextWords9 + " " + rand_sequence[x]

    print('\n '+" ".join(seqOfWords) + ""+toBeAdded + nextWords9)
    ##print(" ".join(seqOfWords) + " " + ' '.join(rand_sequence))
