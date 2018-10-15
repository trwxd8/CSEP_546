import math

class BagOfWordsModel(object):
    """A model that calculates the logsitics regression analysis."""

    def __init__(self):
        self.vocabulary = []
        pass

    def fillVocabulary(self, x):
        cnt = 0
        for example in x:
            curr_words = example.split()
            for word in curr_words:
                if word not in self.vocabulary:
                    cnt += 1
                    #print(word)
                    self.vocabulary.append(word)

    def FrequencyFeatureSelection(self, x, n):

        vocabCount = {}
        for vocab in self.vocabulary:
            vocabCount[vocab] = 0

        #features = []
        for example in x:
            curr_words = example.split()
            #curr_features = []
            for word in curr_words:
                if word in vocabCount:
                    vocabCount[word] += 1
                    #print(word,":", vocabCount[word])
                    #curr_features.append(1)
                #else:
                #    curr_features.append(0)
            #features.append(curr_features)
        #print(features)

        sorted_vocabCount = sorted(vocabCount.items(), key=lambda kv: kv[1], reverse=True)
        print(sorted_vocabCount[:n])
        return sorted_vocabCount

    def MutualInformationFeatureSelection(self, x, y, n):
        cnt = len(y)
        pos_cnt = sum(y)
        neg_cnt = cnt - pos_cnt
        mutualInfo = {}

        for word in self.vocabulary:
            word_cnt = word_pos = word_neg = currMI = 0
            
            for i in range(cnt):
                example = x[i]

                if word in example:
                    word_cnt += 1
                    if y[i] == 1:
                        word_pos += 1
                    else:
                        word_neg += 1

            print(cnt," - pos:", pos_cnt, " neg:",neg_cnt)
            print(word)

            if word_cnt != 0:
                print(word_cnt," - pos:",word_pos, " neg:",word_neg)

                #Calculate P(word,1) and P(word,0)
                prob_word_pos =  (word_pos + 1.0) / (word_cnt + 2.0)
                prob_word_neg = (word_neg + 1.0) / (word_cnt + 2.0)

                currMI += prob_word_pos * math.log2(prob_word_pos / (word_cnt * pos_cnt * 1.0) )
                currMI += prob_word_neg * math.log2(prob_word_neg / (word_cnt * neg_cnt * 1.0) )
            mutualInfo[word] = currMI
            print(word,":",currMI)

        sorted_mutualInfo = sorted(mutualInfo.items(), key=lambda kv: kv[1], reverse=True)
        print(sorted_mutualInfo[:n])
        return sorted_mutualInfo