from HMM import unsupervised_HMM
import pickle


def save_obj(obj, name):
    """
    Save data
    """
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    """
    Load data
    """
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
 
# load data
training = load_obj('./pp_data/training_poem_data')
print ("followig is the data required:")

##print(training[1])

##jfsdhnlksdnojd
# training
for i in range(10, 41, 5):## 41->11
    print ('i ='+str(i))
    HMM = unsupervised_HMM(training, i, 500)
    A = HMM.A
    O = HMM.O

    save_obj(A, './data/transition_matrix_line_'+str(i))
    save_obj(O, './data/observation_matrix_line_'+str(i))
    save_obj(HMM, './data/hmm_line_'+str(i))


