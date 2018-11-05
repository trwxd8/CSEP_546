import math

class BagOfWordsModel(object):
    """A model that calculates the logsitics regression analysis."""

    def __init__(self):
        self.vocabulary = []
        self.preFeaturedWords = []#['call', 'to', 'your']
        pass

    def fillVocabulary(self, x):
         for example in x:
            curr_words = example.split()
            for currWord in curr_words:
                word = currWord.lower()
                #word = ''.join(char.lower() for char in currWord if char.isalnum())
                if word not in self.vocabulary and word not in self.preFeaturedWords:
                    #print(word)
                    self.vocabulary.append(word)

    def LoadFrequencyDictionary(self, x):
        vocabCount = {}
        for vocab in self.vocabulary:
            vocabCount[vocab] = 0

        #features = []
        for example in x:
            curr_words = example.split()
            #curr_features = []
            for currWord in curr_words:
                word = currWord.lower()
                if word in vocabCount:
                    vocabCount[word] += 1

        self.sortedVocabCount = sorted(vocabCount.items(), key=lambda kv: kv[1], reverse=True)

    def FrequencyFeatureSelection(self, n):
        topn_results = self.sortedVocabCount[:n]
        return topn_results

    def LoadMutualInformationDictionary(self, x, y):
        cnt = len(y)
        pos_cnt = sum(y)
        neg_cnt = cnt - pos_cnt
        mutualInfo = {}

        for vocabWord in self.vocabulary:
            word_present = word_pos = word_neg = noword_pos = noword_neg = currMI = 0
            
            for i in range(cnt):
                example = x[i]
                curr_words = example.split()
                vocabContained = False
                for currWord in curr_words:
                    word = currWord.lower()
                    if word == vocabWord:
                        vocabContained = True
                        break

                if vocabContained == True:
                    word_present += 1
                    if y[i] == 1:
                        word_pos += 1
                    else:
                        word_neg += 1
                else:
                    if y[i] == 1:
                        noword_pos += 1
                    else:
                        noword_neg += 1

            word_missing = cnt - word_present

            #Calculate P(word,1) and P(word,0)
            prob_word_pos =  self.prob_func(word_pos, cnt)
            prob_word_neg =  self.prob_func(word_neg, cnt)
            prob_noword_pos =  self.prob_func(noword_pos, cnt)
            prob_noword_neg =  self.prob_func(noword_neg, cnt)
            prob_pos = self.prob_func(pos_cnt, cnt)
            prob_neg = self.prob_func(neg_cnt, cnt)
            prob_word = self.prob_func(word_present, cnt)
            prob_noword = self.prob_func(word_missing, cnt)

            currMI += prob_word_pos * math.log2(prob_word_pos / (prob_word * prob_pos) )
            currMI += prob_word_neg * math.log2(prob_word_neg / (prob_word * prob_neg) )
            currMI += prob_noword_pos * math.log2(prob_noword_pos / (prob_noword * prob_pos) )
            currMI += prob_noword_neg * math.log2(prob_noword_neg / (prob_noword * prob_neg) )
            mutualInfo[vocabWord] = currMI

        self.sortedMutualInfo = sorted(mutualInfo.items(), key=lambda kv: kv[1], reverse=True)

    def MutualInformationFeatureSelection(self, n):
        topn_results = self.sortedMutualInfo[:n]
        return topn_results

    def prob_func(self, observed, total):
        return (observed + 1.0) / (total + 2.0)

    def FeaturizeByWords(self, xTrainRaw, xTestRaw, words):
        # featurize the training data, may want to do multiple passes to count things.
        xTrain = []
        for example in xTrainRaw:
            features = []
            # Have features for a few words\
            lowercase_words = []
            for curr_words in example.split():
                for currWord in curr_words:
                    lowercase_words.append(currWord.lower())

            for key_word in words:
                if key_word in lowercase_words:
                    features.append(1)
                else:
                    features.append(0)
            #print(features)
            xTrain.append(features)

        # now featurize test using any features discovered on the training set. Don't use the test set to influence which features to use.
        xTest = []
        for x in xTestRaw:
            features = []

            # Have features for a few words
            for word in words:
                if word in x:
                    features.append(1)
                else:
                    features.append(0)
            #print(features)
            xTest.append(features)

        return (xTrain, xTest)