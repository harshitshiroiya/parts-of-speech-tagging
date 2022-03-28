###################################
# CS B551 Spring 2021, Assignment #3
#
# Your names and user ids: Lalith Dupathi(ndupathi), Aditya Pandey(adpandey), Kartik Bharadwaj(karbhar)
#
# (Based on skeleton code by D. Crandall)
#
import random
import math
import copy


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    word_count_tot=0
    pos_count_tot=0
    emission_prob={}
    emissions={}
    transition_prob={}
    transitions={}
    priors={}
    start_pos_count={}


    def posterior(self, model, sentence, label):
        if model == "Simple":
            return -999
        elif model == "HMM":
            return -999
        elif model == "Complex":
            return -999
        else:
            print("Unknown algo!")


    '''
    def emission(self,curr_word,curr_pos):
        if curr_word in self.emission_prob and curr_pos not in self.emission_prob[curr_word].keys():
            self.emission_prob[curr_word]={}
            self.emission_prob[curr_word][curr_pos]= 0.00001
            return self.emission_prob[curr_word][curr_pos]

        if curr_word in self.emission_prob:
            return self.emission_prob[curr_word][curr_pos]

        elif curr_word not in self.emission_prob:
            if curr_word not in self.emissions or curr_pos not in self.emissions[curr_word].keys():
                self.emission_prob[curr_word]={}
                self.emission_prob[curr_word][curr_pos]= 0.00001
                return self.emission_prob[curr_word][curr_pos]

            else:
                self.emission_prob[curr_word]={}
                self.emission_prob[curr_word][curr_pos]= self.emissions[curr_word][curr_pos]/ self.transitions[curr_pos]['pos_count']
                return self.emission_prob[curr_word][curr_pos]
    '''


    '''
    The following logic flow was derived from below article to derive emission probabilities
    https://stats.stackexchange.com/questions/212961/calculating-emission-probability-values-for-hidden-markov-model-hmm
    https://github.com/chetan253/B551-Elements-of-AI/blob/master/Part-of-Speech-Tagging/pos_solver.py
    '''

    def emission(self,curr_word,curr_pos):
        if curr_word in self.emission_prob.keys():
            return self.emission_prob[curr_word]
        else:
            if curr_word not in self.emission_prob.keys():
                curr_dict = {}
                for i in ['adv', 'noun', 'adp', 'pron', 'det','.','num', 'prt', 'verb', 'x', 'conj', 'adj']:
                    if curr_word in self.emissions.keys() and i in self.emissions[curr_word].keys():
                        val1= self.emissions[curr_word][i]
                        val2= self.transitions[i]['pos_count']
                        curr_dict[i] = float(val1) / float(val2)
                    elif curr_word in self.emissions.keys() and i not in self.emissions[curr_word].keys():
                        curr_dict[i] = 0.00001
                    else:
                        curr_dict[i]=0.00001
                self.emission_prob[curr_word] = curr_dict
        return self.emission_prob[curr_word]

    '''
    End of logic flow
    '''


    def transition(self,curr_pos):
        if curr_pos in self.transition_prob.keys():
            return self.transition_prob[curr_pos]

        else:
            if curr_pos not in self.transition_prob.keys():
                curr_dict = {}
                for i in ['adv', 'noun', 'adp', 'pron', 'det','.','num', 'prt', 'verb', 'x', 'conj', 'adj']:
                    if curr_pos in self.transitions.keys() and i in self.transitions[curr_pos].keys():
                        val1=self.transitions[curr_pos][i]
                        val2=sum(self.transitions[curr_pos].values())
                        val3=self.transitions[curr_pos]['pos_count']
                        curr_dict[i]=val1/(val2-val3)
                    elif curr_pos in self.transitions.keys() and i not in self.transitions[curr_pos].keys():
                        curr_dict[i]=0.00001
                    else:
                        curr_dict[i] = 0.00001
                self.transition_prob[curr_pos] = curr_dict
        return self.transition_prob[curr_pos]
    

    # Do the training!
    #
    '''
    The concept of storing the count of all the emissions/transitions in the respective dictionary has been implemented after high level
    discussion with peers about the structure of emission and transition dictionaries 
    '''
    def train(self, data):
        for line in data:
            #gives the emissions count of respective word and part of speech
            for i in range(0, len(line[0])):
                word,pos = line[0][i],line[1][i]
                if word != ' ' and word != None:
                    self.word_count_tot+=1
                
                if word not in self.emissions:
                    self.emissions[word]={}
                    self.emissions[word][pos]=0
                    self.emissions[word][pos] = self.emissions[word][pos] +1
                    self.emissions[word]['word_count']=0
                    self.emissions[word]['word_count'] = self.emissions[word]['word_count'] +1

                else:
                    curr_dic = self.emissions[word].keys()
                    if pos in curr_dic:
                        self.emissions[word][pos] = self.emissions[word][pos] +1
                        self.emissions[word]['word_count'] = self.emissions[word]['word_count'] +1
                    else:
                        self.emissions[word][pos] = 1
                        self.emissions[word]['word_count'] = 1


            #gives the transitions count of respective part of speech
            for j in range(0,(len(line[0])-1)):
                current_pos= line[1][j]
                next_pos= line[1][j+1]
                if current_pos != ' ' and current_pos!= None:
                    self.pos_count_tot+=1

                if current_pos not in self.transitions:
                    self.transitions[current_pos]={}
                    self.transitions[current_pos][next_pos]=0
                    self.transitions[current_pos][next_pos]=self.transitions[current_pos][next_pos]+1
                    self.transitions[current_pos]['pos_count'] =0
                    self.transitions[current_pos]['pos_count'] =self.transitions[current_pos]['pos_count'] + 1

                else:
                    curr_dic=self.transitions[current_pos].keys()
                    if next_pos in curr_dic:
                        self.transitions[current_pos][next_pos]=self.transitions[current_pos][next_pos]+1
                        self.transitions[current_pos]['pos_count'] =self.transitions[current_pos]['pos_count'] + 1
                    else:
                        self.transitions[current_pos][next_pos]=1
                        self.transitions[current_pos]['pos_count']=1
            
            last_pos=line[1][-1]
            self.transitions[last_pos]['pos_count']= self.transitions[last_pos]['pos_count']+1

            start_pos=line[1][0]
            if start_pos in self.start_pos_count:
                self.start_pos_count[start_pos]=self.start_pos_count[start_pos]+1
            else:
                self.start_pos_count[start_pos]=1

        for pos in ['adv', 'noun', 'adp', 'pron', 'det', 'num', '.', 'prt', 'verb', 'x', 'conj', 'adj']:
            self.priors[pos]=self.transitions[pos]['pos_count']
            self.priors[pos]=self.priors[pos]/self.pos_count_tot
              

    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    def simplified(self, sentence):
        result=[]
        for word in sentence:
            for pos in ['adv','noun', 'adp', 'pron', 'det', 'num', '.', 'prt', 'verb', 'x', 'conj', 'adj']:
                temp={}
                x=self.emission(word,pos)
                val=max(x.values())
                temp[pos]=float(val)*float(self.priors[pos])
            result.append(max(x,key=x.get))
        return result
    

    '''
    Viterbi was reproduced through the code given in the inclass activity 
    '''
    def hmm_viterbi(self,sentence):
        N= len(sentence)
        V_table = {"adv": [0] * N, "noun" : [0] * N, "adp" : [0] * N, "pron" : [0] * N, "det" : [0] * N, "num" : [0] * N, "." : [0] * N, "prt" : [0] * N, "verb" : [0] * N, "x" : [0] * N, "conj" : [0] * N, "adj" : [0] * N}
        which_table = {"adv": [0] * N, "noun" : [0] * N, "adp" : [0] * N, "pron" : [0] * N, "det" : [0] * N, "num" : [0] * N, "." : [0] * N, "prt" : [0] * N, "verb" : [0] * N, "x" : [0] * N, "conj" : [0] * N, "adj" : [0] * N}
        states=['adv','noun', 'adp', 'pron', 'det', 'num', '.', 'prt', 'verb', 'x', 'conj', 'adj']

        words=[]
        for word in sentence:
            words.append(word)

        for s in states:
            x=self.emission(words[0],s)[s]
            V_table[s][0] = float((self.start_pos_count[s]/sum(self.start_pos_count.values())))*float(x)

        for i in range(1,N):
            for s in states:
                (which_table[s][i], V_table[s][i]) =  max( [ (s0, V_table[s0][i-1] * self.transition(s0)[s]) for s0 in states ], key=lambda l:l[1] ) 
                V_table[s][i] *= self.emission(words[i],s)[s]

        last_seq=[""]*N
        pos_last_value=[]
        for s0 in states:
            pos_last_value.append(V_table[s0][-1])
        index=pos_last_value.index(max(pos_last_value))
        last_seq[N-1]=states[index]   
        for i in range(N-2, -1, -1):
            last_seq[i] = which_table[last_seq[i+1]][i+1]
        return last_seq    

    

    def generate_sample(self,sentence,sample):
        N=len(sentence)
        tags=['adv','noun', 'adp', 'pron', 'det', 'num', '.', 'prt', 'verb', 'x', 'conj', 'adj']
        for index in range(N):
            curr_word= sentence[index]
            probs=[0]*12
        
            s_1 = sample[index - 2] if index > 1 else " "
            s_0 = sample[index - 1] if index > 0 else " "

            for j in range(12):
                s_2=tags[j]
                emission1= self.emission(curr_word,s_2)[s_2]
                transition1= self.transition(s_0)[s_2]
                transition2=self.transition(s_1)[s_2]
                
                if index==0:
                    probs[j]= emission1*(self.start_pos_count[s_2]/sum(self.start_pos_count.values()))
                elif index==1:
                    probs[j]= emission1*transition2
                else:
                    probs[j]=emission1*transition1*transition2

            tags_list=['adv','noun', 'adp', 'pron', 'det', 'num', '.', 'prt', 'verb', 'x', 'conj', 'adj']
            i= probs.index(max(probs))
            final_tag= tags_list[i]
            sample[index]=final_tag
        
        return sample


    def complex_mcmc(self, sentence):
        N= len(sentence)
        samples_1000= []
        initial_sample=['adv']*N
        final=[]
        samples_1000.append(initial_sample)
        for i in range(1,800):
            samples_1000.append(self.generate_sample(sentence,copy.deepcopy(samples_1000[i-1])))
            
        for j in range(N):
            pos_tags1=[]
            for i in range(400,len(samples_1000)):
                pos_tags1.append(samples_1000[i][j])
            final.append(max(pos_tags1,key=pos_tags1.count))

        return final

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        else:
            print("Unknown algo!")

