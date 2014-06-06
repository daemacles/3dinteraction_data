import pylab as pl
import matplotlib.pyplot as plt
from collections import Counter
import itertools

# All functions generically take in a dataset 'a' and an optional grouping
# indicator 'groups'
#
# a - should be a list of lists, one for each set of samples for the condition
#     of interest (e.g. subject/input/output)
# groups - should be a list of natural numbers the same length as a. All sets
#          of samples with the same group will be treated as a single condition
#          for the purposes of calculating summary statistics (including drawing
#          boxplots)

# Utility function: Transform a list of scenes into a list of times
def scenes2times(l):
    return [l[i].endTime - l[i].startTime for i in range(len(l))]

# Utility function: Split the input dataset further by subject
def splitjoinbysubject(a):
    ret = []
    for l in a:
        subjects = set([x.subject for x in l])
        for s in subjects:
            newlist = [x for x in l if x.subject == s]
            ret = ret + [newlist]
    return ret

# Utility function: reaggregates grouped data
def regroup(a,groups):
    if groups is not None:
        groupcounts = Counter(groups)
        data = []
        for i in groupcounts.keys():
            newlist = []
            for g,l in zip(groups,a):
                if g == i: newlist.extend(l)
            data.append(newlist)
    else:
        data = a
    return data

# Print out summary statistics for dataset
def distributionstats(a,groups=None,labels=None):
    print "Timing Stats: mean std min max | quartile1 median quartile3"
    for (idx,l) in list(enumerate(regroup(a,groups))):
        s = ''
        if labels is not None: s = '%s: '%labels[idx]
        print "{:10}\t{}\t{}\t{}\t{}\t|\t{}\t{}\t{}".format(s, mean(l), std(l), min(l), max(l),percentile(l,25), median(l),percentile(l,75))

# Helper function - boxplots data in grouped form
def plotgroupeddistributions(a,groups,labels=None):
    margin = 0.35
    groupcounts = Counter(groups)
    counts = {x:0 for x in groupcounts.keys()}

    for (idx,l) in list(enumerate(a,start=1)):
        group = groups[idx-1]
        xpos = float(counts[group])/(groupcounts[group]-1)*(1-2*margin)+group-(0.5-margin)
        counts[group] += 1
        plot(ones(len(l))*xpos, l, '.')

    return boxplot(regroup(a,groups),False,'')

# Constructs a boxplot of the data and also plots each individual data point
def plotdistributions(a,groups=None,labels=None):
    fig, axes = subplots()
    axes.set_ylabel('Completion Time (s)')
    if labels is not None:
        xtickNames = plt.setp(axes, xticklabels=labels)

    if groups is not None:
        n = groups.count(1)
        axes.set_color_cycle([axes._get_lines.color_cycle.next() for i in range(n)])
        return plotgroupeddistributions(a,groups,labels)
    else:
        for (idx,l) in list(enumerate(a,start=1)):
            plot(ones(len(l))*idx, l, '.')
        return boxplot(a,False,'')

# Constructs histogram of the data
def histogramdistributions(a,cutoff=40,numbins=30,labels=None):
    figure()
    n = len(a)
    for (idx,l) in list(enumerate(a,start=1)):
        l = np.array(l)
        axes = subplot(n, 1, idx)
        hist(l[l < cutoff], bins=numbins, range=(0,cutoff))
        if labels is not None:
            title(labels[idx])


def main():
    groups = [1]*10 + [2]*10 + [3]*10
    leap = getScenes('input is "leap" and output is "3dHeadtracked"')
    mousekbd = getScenes('input is "mousekbd" and output is "3dHeadtracked"')
    hydra = getScenes('input is "3dmouse" and output is "3dHeadtracked"')
    data = map(scenes2times, [leap, mousekbd, hydra])
    groupeddata = map(scenes2times, splitjoinbysubject([leap, mousekbd, hydra]))

