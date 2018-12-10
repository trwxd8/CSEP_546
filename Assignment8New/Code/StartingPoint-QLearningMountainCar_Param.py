
import gym

env = gym.make('MountainCar-v0')

import random
import QLearning # your implementation goes here...
import Assignment7Support
import math

discountRate = 0.98          # Controls the discount rate for future rewards -- this is gamma from 13.10
actionProbabilityBase = 1.8  # This is k from the P(a_i|s) expression from section 13.3.5 and influences how random exploration is
randomActionRate = 0.01      # Percent of time the next action selected by GetAction is totally random
learningRateScale = 0.01     # Should be multiplied by visits_n from 13.11.
trainingIterations = 20000

#discountRateSet = [discountRate]
#actionProbabilitySet = [math.e]
#trainingIterationsCount = [20000]

discountRateSet = [.5, .75, .9, .95, 1]
actionProbabilitySet = [.75, 1.5, 2.2, 2.7, 3]
trainingIterationsCount = [10000, 15000, 20000, 25000, 30000]


for currDiscountRate in discountRateSet:
    for currActionProb in actionProbabilitySet:
        textFile = "disc_"+str(currDiscountRate)+"_actionbase_"+str(currActionProb)+".txt"
        with open(textFile, "a") as currFile:
            for currIterCount in trainingIterationsCount:
                print("\nDiscount Rate:"+str(currDiscountRate)+" Action Probability Base:"+str(currActionProb)+" Iteration Count:"+str(currIterCount)+"\n")               

                currFile.write("\nDiscount Rate:"+str(currDiscountRate)+" Action Probability Base:"+str(currActionProb)+" Iteration Count:"+str(currIterCount)+"\n")               
                qlearner = QLearning.QLearning(stateSpaceShape=Assignment7Support.MountainCarStateSpaceShape(), numActions=env.action_space.n, discountRate=currDiscountRate)

                for trialNumber in range(currIterCount):
                    observation = env.reset()
                    reward = 0
                    for i in range(201):
                        #env.render()

                        currentState = Assignment7Support.MountainCarObservationToStateSpace(observation)
                        action = qlearner.GetAction(currentState, False, learningMode=True, randomActionRate=randomActionRate, actionProbabilityBase=currActionProb)

                        oldState = Assignment7Support.MountainCarObservationToStateSpace(observation)
                        observation, reward, isDone, info = env.step(action)
                        newState = Assignment7Support.MountainCarObservationToStateSpace(observation)

                        # learning rate scale 
                        qlearner.ObserveAction(oldState, action, newState, reward, learningRateScale=learningRateScale)

                        if isDone:
                            if(trialNumber%1000) == 0:
                                currFile.write(str(trialNumber)+":" +str(i) + ":" +str(reward)+"\n")
                            break

                ## Now do the best n runs I can
                #input("Enter to continue...")

                n = 20
                totalRewards = []
                for runNumber in range(n):
                    observation = env.reset()
                    totalReward = 0
                    reward = 0
                    for i in range(201):
                        #renderDone = env.render()

                        currentState = Assignment7Support.MountainCarObservationToStateSpace(observation)
                        observation, reward, isDone, info = env.step(qlearner.GetAction(currentState, False, learningMode=False, randomActionRate=randomActionRate, actionProbabilityBase=currActionProb))

                        totalReward += reward

                        if isDone:
                            #renderDone = env.render()
                            currFile.write(str(i) + ":" +str(reward)+"\n")
                            totalRewards.append(totalReward)
                            break

                currFile.write(str(totalRewards) + "\n")
                currFile.write("Your score:" + str(sum(totalRewards) / float(len(totalRewards)))+"\n")