import pylab as pl
import matplotlib.pyplot as plt
import itertools

subjects = ["{:02}".format(i) for i in xrange(2,13) if i != 11]

def distanceFromEnd(scene):
    P = scene.inputs[:,1:4]
    # the scene stores the original start and end pos, so need to figure out
    # where on the x-axis the end would be
    end = norm(scene.endPos - scene.startPos) * np.array([1,0,0])
    D = [norm(end-p) for p in P]
    return D

def genDistanceHistogram(input, output):
    scenes = getScenes('(subject is "02" or subject is "03") and input is "{}" and output is "{}"'.format(input, output))
    D = np.array(list(itertools.chain(*map(distanceFromEnd, scenes))))
    D = D[D < 35]               # filter out outliers
    plt.title("{} - {}".format(input, output))
    plt.hist(D, bins=20, normed=True)
        
def genDistanceHistograms():
    f = pl.figure()
    i = 1
    for output in ['2dprojections', '3dHeadtracked']:
        for input in ['mousekbd', '3dmouse', 'leap']:
            axes = pl.subplot(2, 3, i)
            genDistanceHistogram(input, output)
            i += 1
    plt.suptitle("Distance from End Position", fontsize=16)
    return f

def extraDistanceTraveled(scene):
    P = scene.inputs[:,1:4]
    # the scene stores the original start and end pos, so need to figure out
    # where on the x-axis the end would be
    min_d = norm(scene.endPos - scene.startPos)

    # integrate distance traveled over scene points
    D = P[1:] - P[:-1]          # discrete differential
    actual_d = np.sum([norm(p) for p in D])
    return actual_d / min_d

def extraDistanceStats():
    print "Extra Distance Stats:  mean std min max"
    for output in ['2dprojections', '3dHeadtracked']:
        for input in ['mousekbd', '3dmouse', 'leap']:
            scenes = getScenes('input is "{}" and output is "{}"'.format(input, output))
            eds = np.array(map(extraDistanceTraveled, scenes))            
            eds = eds[eds < 20] # exclude outliers
            print "{:8} {:13}:\t{}\t{}\t{}\t{}".format(input, output, mean(eds), std(eds), min(eds), max(eds))

def improvement(subject, input, output):
    scenes = getScenes('subject is "{}" and input is "{}" and output is "{}" order by startTime'.format(subject, input, output))
    N = 3
    first = mean(map(extraDistanceTraveled, scenes[:N]))
    last = mean(map(extraDistanceTraveled, scenes[-N:]))
    return first - last

def improvementStats():
    print "Improvement Stats:  mean std min max"
    for output in ['2dprojections', '3dHeadtracked']:
        for input in ['mousekbd', '3dmouse', 'leap']:
            imps = [improvement(s, input, output) for s in subjects]
            print "{:8} {:13}:\t{}\t{}\t{}\t{}".format(input, output, mean(imps), std(imps), min(imps), max(imps))


            
def main():
    #populateDB()
    scenes = getScenes('(subject is "05" or subject is "04") and input is "leap" and output is "3dHeadtracked"')
    map(cylinderCompress, scenes)
    f = figure(bgcolor=(0,0,0))
    _ = map(lambda s: plotScene(s, colormap = 'spring', tube_sides=3), scenes)

if __name__ == "__main__":
    #extraDistanceStats()
    #print
    #improvementStats()
    pass

