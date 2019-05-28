class Training_State_Manager(object):


    def __init__(self,trainer):
        self.epoch = None
        self.iter  = None
        self.bind(trainer)



    def bind(self,trainer):
        trainer.stateFcn = self.stateFcn



    def stateFcn(self,input,output,loss,epoch,iteration,t):
        e = epoch[1:3]
        i = iteration
        g = (e[0]*i[1]+i[0],e[1]*i[1])
        if not e[0]==self.epoch:
            self.epoch = e[0]
            self.epochFcn(input,output,loss,epoch,iteration,t)
        if not i[0]==self.iter:
            self.iter = i[0]
            self.iterationFcn(input,output,loss,epoch,iteration,t)
        if g[0]==g[1]-1:
            self.endFcn(input,output,loss,epoch,iteration,t)
        return



    def iterationFcn(self,input,output,loss,epoch,iteration,t):
        return



    def epochFcn(self,input,output,loss,epoch,iteration,t):
        return



    def endFcn(self,input,output,loss,epoch,iteration,t):
        return
