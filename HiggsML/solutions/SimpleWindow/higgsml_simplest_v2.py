print "This script computes the submission file with just a simple window on one variable DER_mass_MMC"
# completely self contained
# v2 20140522 : final ordering was buggy and too complicated. Simplified considerably

import csv

print "Reading in training file"
alltraining = list(csv.reader(open("training.csv","rb"), delimiter=','))

# first line is the list of variables
headertraining        = alltraining[0]
# cut off first line
alltraining=alltraining[1:]

#ditto for test file
print "Reading in test file"
alltest = list(csv.reader(open("test.csv","rb"), delimiter=','))
headertest        = alltest[0]
alltest=alltest[1:]

# get the index of a few variables
immc=headertraining.index("DER_mass_MMC")
injet=headertraining.index("PRI_jet_num")
iweight=headertraining.index("Weight")
ilabel=headertraining.index("Label")
iid=headertraining.index("EventId")


print "Loop on training dataset and compute the score"

headertraining+=["myscore"]
for entry in alltraining:
    # turn all entries from string to float, except EventId and PRI_jet_num to int, except label remains string
    for i in range(len(entry)):
        if not i in [ilabel,iid,injet]:
            entry[i]=float(entry[i])
        if i in [iid,injet]:
            entry[i]=int(entry[i])
            
            
    myscore=-abs(entry[immc]-125.) # this is a simple discriminating variable. Signal should be closer to zero.
                                   # minus sign so that signal has the highest values
                                   # so we will be making a simple window cut on the Higgs mass estimator
                                   # 125 GeV is the middle of the window
    entry+=[myscore]
    
# at this stage alltraining is a list (one entry per line) of list of variables
# which can be conveniently accessed by getting the index from the header 

threshold=-22 # somewhat arbitrary value, should be optimised

print "Loop again to determine the AMS, using threshold:",threshold
sumsig=0.
sumbkg=0.
iscore=headertraining.index("myscore")
for entry in alltraining:
    myscore=entry[iscore]
    entry+=[myscore]
    weight=entry[iweight]
    # sum event weight passing the selection. Of course in real life the threshold should be optimised
    if myscore >threshold:
        if entry[ilabel]=="s":
            sumsig+=weight
        else:
            sumbkg+=weight    

# ok now we have our signal (sumsig) and background (sumbkg) estimation

# compute AMS
def ams(s,b):
    from math import sqrt,log
    if b==0:
        return 0

    return sqrt(2*((s+b+10)*log(1+float(s)/(b+10))-s))

print " AMS computed from training file :",ams(sumsig,sumbkg),"( signal=",sumsig," bkg=",sumbkg,")"

print "Compute the score for the test file entries "

#recompute variable indices for safety 
immc=headertest.index("DER_mass_MMC")
injet=headertest.index("PRI_jet_num")
iid=headertest.index("EventId")
headertest+=["myscore"]

for entry in alltest:
    # turn all entries from string to float, except EventId and PRI_jet_num to int (there is no label)
    for i in range(len(entry)):
        if not i in [iid,injet]:
            entry[i]=float(entry[i])
        else:    
            entry[i]=int(entry[i])
    # add my score
    myscore=-abs(entry[immc]-125.)                                    
    entry+=[myscore]





iscore=headertest.index("myscore")
if iscore<0:
    print "ERROR could not find variable myscore"
    raise Exception # should not happen



print "Sort on the score " 
# in the first version of the file, an auxilliary map was used, but this was useless
alltestsorted=sorted(alltest,key=lambda entry: entry[iscore])
# the RankOrder we want is now simply the entry number

submissionfilename="submission_simplest.csv"
print  "Final loop to write the submission file",submissionfilename
outputfile=open(submissionfilename,"w")
outputfile.write("EventId,RankOrder,Class\n")
iid=headertest.index("EventId")
if iid<0:
    print "ERROR could not find variable EventId in test file"
    raise Exception # should not happen

rank=1 # kaggle wants to start at 1
for entry in alltestsorted:
    # compute label 
    slabel="b"
    if entry[iscore]>threshold: # arbitrary threshold
        slabel="s"

    outputfile.write(str(entry[iid])+",")
    outputfile.write(str(rank)+",")
    outputfile.write(slabel)            
    outputfile.write("\n")
    rank+=1


outputfile.close()
print " You can now submit ",submissionfilename," to kaggle site"



# delete big objects
del alltraining,alltest,alltestsorted

