import nltk
import random
import string

text_file = open ("dataset2.txt","r",encoding="utf-8")
article_text=text_file.read()
text_file.close()

article_text=article_text.lower()
article_text= article_text.translate(str.maketrans('', '', string.punctuation))


ngrams = {}
words = 3

words_tokens = nltk.word_tokenize(article_text)
for i in range(len(words_tokens)-words):
    seq = ' '.join(words_tokens[i:i+words])
    if  seq not in ngrams.keys():
        ngrams[seq] = []
    ngrams[seq].append(words_tokens[i+words])

print ("ngrams dictionary")
for key, value in ngrams.items() :
    print (key, value,)



for m in range (5):
    number= random.randint(1,len(words_tokens)-words)
    curr_sequence = ' '.join(words_tokens[number:number+words])
    output = curr_sequence

    for i in range(100):
        if curr_sequence not in ngrams.keys():
            break
        possible_words = ngrams[curr_sequence]
        next_word = possible_words[random.randrange(len(possible_words))]
        output += ' ' + next_word
        seq_words = nltk.word_tokenize(output)
        curr_sequence = ' '.join(seq_words[len(seq_words)-words:len(seq_words)])

    print(output,"\n")