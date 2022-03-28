###################################
# CS B551 Spring 2021, Assignment #3
#
# Your names and user ids:
#
# (Based on skeleton code by D. Crandall)
#


import random
import sys
import os
import math
from collections import Counter
from pos_scorer import Score

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#

class Solver:

    position_count = {}
    word_count = {}
    word_location = {}
    word_location_probablity = {}
    transition_probablity = {}
    emission_probablity = {}
    p_word_pos = {}
    temp_word = {}
    start_probablity = {}
    
   
    def posterior(self, model, sentence, label):
        if model == "Simple":
            return -999
        elif model == "HMM":
            return -999
        elif model == "Complex":
            return -999
        else:
            print("Unknown algo!")

        

    def train(self, data):
        
       
        self.word_location = {'adj' : [], 'adv' : [], 'adp' : [], 'conj' : [], 'det' : [], 'noun' : [], 'num' : [], 'pron' : [], 'prt' : [], 'verb' : [], 'x' : [], '.' : []}
        
        self.start_probablity = {}
        self.start_probablity.update(dict.fromkeys(['adj' ,'adv' ,'adp' ,'conj','det' ,'noun','num' ,'pron','prt' ,'verb','x'  ,'.'], 0 ))

        self.transition_probablity = { }
        
        self.position_count = {}
        self.position_count.update(dict.fromkeys(['adj' ,'adv' ,'adp' ,'conj','det' ,'noun','num' ,'pron','prt' ,'verb','x'  ,'.'], 0 ))
        c = 0
        for component in data :
            component_len = len(component[ 1 ])
            if component_len > 1 :
                for i in range( component_len - 1)   :
                    previous = component[ 1 ][ i ]
                    next_component = component[ 1 ][ i + 1 ]
                    self.start_probablity[ previous ] = self.start_probablity[ previous ] + 1
                    if ( previous , next_component ) not in self.transition_probablity:
                        self.transition_probablity.update( { ( previous , next_component ) : 1 } )
                    else:
                        self.transition_probablity[ ( previous , next_component ) ] = self.transition_probablity[ ( previous , next_component ) ] + 1
                    self.word_location[ component[ 1 ][ i ] ].append( component[ 0 ][ i ] )
                self.word_location[ component[ 1 ][ i + 1 ] ].append( component[ 0 ][ i + 1 ] )

        for pos in self.position_count :
             for word in self.word_count :
                self.position_count[ pos ] = self.position_count[ pos ] / c
                self.word_count[ word ] = self.word_count[ word ] / c
        

        
        for component in data:
            for pos in component[ 1 ]:
                for word in component[ 0 ] :

                    self.position_count[ pos ] = self.position_count[ pos ] + 1
                    c = c + 1
                
                    if word in self.word_count :
                        self.word_count[word] = self.word_count[word] + 1
                    else:
                        self.word_count[ word ] = 1
        
        
                
        
        value = 0
        for i in self.start_probablity:
            value = value + self.start_probablity[ i ] 


        for component in self.start_probablity.keys( ) :
            self.start_probablity[ component ] =  self.start_probablity[ component ] / value
        
        for component in data :
            component_len1 = len( component[ 0 ] )
            for i in range( component_len1 ) :
                if component[ 0 ][ i ] not in self.p_word_pos :    
                    self.p_word_pos[ component[ 0 ][ i ] ] = [ component[ 1 ][ i ] ]
                else :
                    self.p_word_pos[ component[ 0 ][ i ] ].append( component[ 1 ][ i ] )
        total_value = 0    
        for i in self.transition_probablity:
            total_value = total_value + self.transition_probablity[ i ] 
        
        for i in self.transition_probablity:
            self.transition_probablity[ i ] = self.transition_probablity[ i ] / total_value  
        self.emission_probablity = { }
        
        for pos, words in self.word_location.items():
            c1 = Counter(words)
            c1 = Counter (th for th in c1.elements())
        
        for pos, words in self.word_location.items():
            value = 0
            c1= Counter(words)
            c1 = Counter (th for th in c1.elements())
            for component, co in c1.items( ) :
                value = value + co
            for component, co in c1.items( ) :
                self.emission_probablity.update( { ( component , pos ) :  co / value } )
        
        self.word_location_probablity = { }
        
        for word , pos in self.p_word_pos.items():
            value = 0
            c1 = Counter( pos )
            c1 = Counter ( th for th in c1.elements( ) )
            for component , co in c1.items( ) :
                value = value + co
            for component, co in c1.items( ) :
                if word not in self.word_location_probablity :
                    self.word_location_probablity.update( { word : [ [ co / value , component ] ] } )
                else :
                    self.word_location_probablity[ word ].append( [ co / value , component ] )
        
        parts=['adj' ,'adv' ,'adp' ,'conj','det' ,'noun','num' ,'pron','prt' ,'verb','x'  ,'.' ]
        
        
        temp=[]
        for i in parts:
            for j in parts:
                temp.append((i,j))
        
        l=list(self.transition_probablity)
        
        difference=list(set(temp)-set(l))
        transition_min = min( self.transition_probablity.values( ) )
        for i in difference :
            self.transition_probablity[i] = transition_min / 20
        pass



    def simplified(self, sentence):       
        answer_list = [ ]
        sentence_list = list(sentence)
        for word in sentence_list :
            if word in self.p_word_pos :
                temp = [ ]
                for component in self.word_location_probablity[ word ] :
                    temp.append( component[ 0 ] )
                answer_list.append( self.word_location_probablity[ word ][ temp.index( max( temp ) ) ][ 1 ] )
            else :
                answer_list.append( 'x' )
        return answer_list



    def hmm_viterbi(self, sentence):
        curr_dist ={'adj' : [], 'adv' : [], 'adp' : [], 'conj' : [], 'det' : [], 'noun' : [], 'num' : [], 'pron' : [], 'prt' : [], 'verb' : [], 'x' : [], '.' : []}
        
        states = [ 'adj' ,'adv' ,'adp' ,'conj' ,'det' ,'noun' , 'num' , 'pron' ,'prt' ,'verb' ,'x','.' ]
        
        answer_list = [ ]
        previous_pos = ''
        
        
        figures = {}
        figures.update(dict.fromkeys(['adj' ,'adv' ,'adp' ,'conj','det' ,'noun','num' ,'pron','prt' ,'verb','x'  ,'.'], None ))
        sentence_list = list( sentence )
        word = sentence_list[0]
        p = 0
        for pos in states :
            if ( word , pos ) in self.emission_probablity :
                m = self.emission_probablity[ ( word , pos ) ]
                n = self.start_probablity[ pos ]
                p = m * n
                curr_dist[ pos ].append(("noun",p))
            else :
                curr_dist[ pos ].append( ("noun",1e-4))
        
        answer_list.append( previous_pos )
        
        for word in sentence_list[ 1: ] :
            t = figures
            for pos in states :
                maximum_probablity = 0
                for previous_pos , previous_prob in curr_dist.items( ) :
                    if ( word , pos ) in self.emission_probablity :
                        temp = previous_prob[ -1 ][ 1 ] * self.emission_probablity[ ( word , pos ) ] * self.transition_probablity[ ( previous_pos , pos ) ]
                        if maximum_probablity < temp :
                            maximum_probablity = temp
                            new_pos = previous_pos
                if maximum_probablity == 0 :
                    for previous_pos , previous_prob in curr_dist.items( ) :
                        temp = previous_prob[ -1 ][ 1 ] * self.transition_probablity[ ( previous_pos , pos ) ] * 5e-7
                        if maximum_probablity < temp :
                            maximum_probablity = temp
                            new_pos = previous_pos           
                t[ pos ] = ( new_pos , maximum_probablity )
            for i , j in t.items( ) :
                curr_dist[ i ].append( j )
      
        # Backtracking
      
        maximum_probablity = 0
        t_pos = ""
        temp = ""
        for (pos , word) in curr_dist.items( ) :
            if(maximum_probablity < word[ -1 ][ 1 ]) :
                maximum_probablity = word[ -1 ][ 1 ]
                t_pos = word[ -1 ][ 0 ]
                temp = pos

        maximum_probablity = 0
        answer_list = [ ]
        if not temp :
            temp = 'noun'
        answer_list.append( temp )
        if not t_pos :
            maximum_probablity = 0
            for pos in states :
                if maximum_probablity < self.transition_probablity[ ( answer_list[ -1 ] , pos ) ] :
                    maximum_probablity = self.transition_probablity[ ( answer_list[ -1 ] , pos ) ]
                    t_pos = pos
        
        answer_list.append( t_pos )
        
        
        sentence_len = len(sentence)
        for i in range( sentence_len - 2 , 0 , -1 ) :
            if t_pos :
                t_pos = curr_dist[ t_pos ][ i ][ 0 ]
            else :
                maximum_probablity = 0.0
                for pos in states :
                    if maximum_probablity < self.transition_probablity[ answer_list[ -1 ] ][ pos ] :
                        maximum_probablity = self.transition_probablity[ answer_list[ -1 ] ][ pos ]
                        t_pos = pos

            answer_list.append( t_pos )
        answer_list.reverse( )
        return answer_list[ :sentence_len ]




    def complex_mcmc(self, sentence):

            answer_list = [ "noun" ] * len( sentence )
            states = [ 'adj' ,'adv' ,'adp' ,'conj' ,'det' ,'noun' , 'num' , 'pron' ,'prt' ,'verb' ,'x','.' ]
            bias_prob = [ ]
            sentence_list = list( sentence )
            for word in sentence_list:
                temp = [ ]
                for pos in states :
                    if ( word , pos ) in self.emission_probablity :           
                        temp.append( self.emission_probablity[ ( word , pos ) ] )
                    else :
                        temp.append( 2 )

                min_temp = min( temp )
                
                for i in range( len( temp ) ) :
                    if temp[ i ] == 2 :
                        temp[ i ] = min_temp * 1e-24
                total_value = sum( temp )
                
                for i in range( len( temp ) ) :
                    temp[ i ] /= total_value
                x1 = 0
                for i in range( len( temp ) ) :
                    x1 += temp[ i ]
                    temp[ i ] = x1
                bias_prob.append( temp )

            answer_list = [ ]
        
            l = 0
            for l in range(len( sentence )) :
                k = 1000
                answer_list.append( [ ] )
                while( k > 0 ) :
                    k -= 1
                    ran = random.random()
                    for i in range( 12 ) :
                        if( ran <= bias_prob[ l ][ i ] ) :
                            answer_list[ l ].append( states[ i ] )
                            break
            
            solution = [ ]
            for component in answer_list :
                solution.append( component[ -1 ] )
            return solution




    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")
