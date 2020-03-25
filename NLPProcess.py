import stanfordnlp
import numpy as np
def NLPProcess(druglist,df_interaction):
    def addMechanism(node):
        if int(sonsNum[int(node-1)])==0:
            return
        else:
            for k in sons[node-1]:
                if int(k)==0:
                    break
                if dependency[int(k - 1)].text == drugA[i] or dependency[int(k - 1)].text == drugB[i]:
                    continue
                quene.append(int(k))
                addMechanism(int(k))
        return quene

    nlp = stanfordnlp.Pipeline()
    event=df_interaction['interaction']
    mechanism=[]
    action=[]
    drugA=[]
    drugB=[]
    for i in range(len(event)):
        doc=nlp(event[i])
        dependency = []
        for j in range(len(doc.sentences[0].words)):
            dependency.append(doc.sentences[0].words[j])
        sons=np.zeros((len(dependency),len(dependency)))
        sonsNum=np.zeros(len(dependency))
        flag=False
        count=0
        for j in dependency:
            if j.dependency_relation=='root':
                root=int(j.index)
                action.append(j.lemma)
            if j.text in druglist:
                if count<2:
                    if flag==True:
                        drugB.append(j.text)
                        count+=1
                    else:
                        drugA.append(j.text)
                        flag=True
                        count+=1
            sonsNum[j.governor-1]+=1
            sons[j.governor-1,int(sonsNum[j.governor-1]-1)]=int(j.index)
        quene=[]
        for j in range(int(sonsNum[root-1])):
            if dependency[int(sons[root-1,j]-1)].dependency_relation=='obj' or dependency[int(sons[root-1,j]-1)].dependency_relation=='nsubj:pass':
                quene.append(int(sons[root-1,j]))
                break
        quene=addMechanism(quene[0])
        quene.sort()
        mechanism.append(" ".join(dependency[j-1].text for j in quene))
        if mechanism[i]=="the fluid retaining activities":
            mechanism[i]="the fluid"
        if mechanism[i]=="atrioventricular blocking ( AV block )":
            mechanism[i]='the atrioventricular blocking ( AV block ) activities increase'
    return mechanism,action,drugA,drugB