

class mAPHolder:
    bestmAP = 0
    def __init__(self, earlyStopCount : int):
        self.mAPs = []
        self.earlyStopCount = earlyStopCount
        self.bestmAP = 0
    
    def getBestmAP(self):
        return self.bestmAP
    def setBestmAP(self, bestmAP : float):
        self.bestmAP = bestmAP

    def addmAP(self, mAP : float):
        if (len(self.mAPs) >= self.earlyStopCount):
            self.mAPs.pop(0) #remove first element
        self.mAPs.append(mAP)
    
    def checkIfShouldEarlyStop(self, mAP : float):
        if (len(self.mAPs) >= self.earlyStopCount):
            if (min(self.mAPs) > mAP): #if worst mAP from previous mAPs is better than now 
                print("EARLY STOPPING")
                return True
        return False
