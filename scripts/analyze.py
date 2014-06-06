import pylab as pl
import matplotlib.pyplot as plt
import itertools

subjects = ["{:02}".format(i) for i in xrange(2,14) if i != 11]

def distanceFromEnd(scene):
    P = scene.inputs[:,1:4]
    # the scene stores the original start and end pos, so need to figure out
    # where on the x-axis the end would be
    end = norm(scene.endPos - scene.startPos) * np.array([1,0,0])
    D = np.array([norm(end-p)/norm(end) for p in P])
    D = D[D < 1.5]
    return D

def genDistanceHistogram(input, output):
    scenes = getScenes('input is "{}" and output is "{}"'.format(input, output))
    D = np.array(list(itertools.chain(*map(distanceFromEnd, scenes))))
    D = D[D < 35]               # filter out outliers
    plt.title("{} - {} ({:.2f}s)".format(input, output, sum([s.numInputs for s in scenes])/30.0))
    plt.hist(D, bins=20, normed=True, color='red')
        
def genDistanceHistograms():
    f = pl.figure(figsize=(16,9))
    i = 1
    for output in ['2dprojections', '3dHeadtracked']:
        for input in ['mousekbd', '3dmouse', 'leap']:
            axes = f.add_subplot(2, 3, i, axisbg='black')
            genDistanceHistogram(input, output)
            i += 1
    plt.suptitle("Relative Distance from End Position", fontsize=16)
    f.subplots_adjust(hspace=0.2, wspace=0.17, top=0.91, right=0.97, bottom=0.07, left=0.05)
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


def genDurationHistogram(input, output):
    scenes = getScenes('input is "{}" and output is "{}"'.format(input, output))
    D = np.array([s.endTime - s.startTime for s in scenes])
    D = D[D < 30]               # ignore runs longer than 30 seconds
    Y,X = np.histogram(D, bins = 20)
    Xn = (X[:-1] + X[1:])/2.0
    pl.plot(Xn, Y, label="{} - {}".format(input, output))
    #plt.hist(D, bins=20, histtype="step")#normed=True)

def genDurationHistograms():
    f = pl.figure()
    i = 1
    for output in ['2dprojections', '3dHeadtracked']:
        for input in ['mousekbd', '3dmouse', 'leap']:
            #axes = pl.subplot(2, 3, i)
            genDurationHistogram(input, output)
            i += 1
    plt.suptitle("Duration", fontsize=16)
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Trials")
    return f

def getAverageTrail(input, output):
    scenes = getScenes('input is "{}" and output is "{}"'.format(input, output))
    map(cylinderCompress, scenes)
    inputs = [s.inputs for s in scenes]
    start = -5
    end = 40
    divs = 10
    vals = [list() for i in xrange((end-start)*divs)]
    for input in inputs:
        for p in input[:,1:3]:
            cell = int((p[0] - start) * divs)
            vals[cell].append(p[1])
    for l in vals:
        if len(l) == 0: l.append(0)
    return vals
    '''
    X = np.zeros((len(inputs), max(map(len, inputs))))
    Y = X.copy()
    for i in xrange(len(inputs)):
        x = inputs[i][:,1]
        y = inputs[i][:,2]
        X[i,:len(x)] = x
        Y[i,:len(y)] = y
    Xa = mean(X,0)
    Ya = mean(Y,0)
    return Xa, Ya
    '''

def genHeatmap(input, output):
    scenes = getScenes('input is "{}" and output is "{}"'.format(input, output))
    map(cylinderCompress, scenes)
    for scene in scenes:
        duration = scene.endTime - scene.startTime
        speed = 1 - min(duration/10, 1)
        speed = int(duration < 10)
        plt.title("{} - {}".format(input, output))
        plt.axis((-5, 45, 0, 35))
        plt.xlabel("Distance (cm)")
        plt.ylabel("Distance (cm)")
        pl.plot(scene.inputs[:,1], scene.inputs[:,2], color=(1-speed,speed,0,.2), linewidth=1)

def genHeatmaps():
    f = pl.figure(figsize=(16,9))
    i = 1
    for output in ['2dprojections', '3dHeadtracked']:
        for input in ['mousekbd', '3dmouse', 'leap']:
            axes = f.add_subplot(2, 3, i, axisbg='black')
            genHeatmap(input, output)
            i += 1
    plt.suptitle("Path", fontsize=16)
    plt.legend()
    f.subplots_adjust(hspace=0.2, wspace=0.17, top=0.91, right=0.97, bottom=0.07, left=0.05)
    return f

def fancyPlot(input, output):
    '''
    Plots each user restricted to a slice of the cylinder
    '''
    num = len(subjects)
    i = 0
    rads_per_subject = np.pi*2 / num
    f = figure(bgcolor=(0,0,0))
    for subject in subjects:
        start = i * rads_per_subject
        end = (i+1) * rads_per_subject
        point_color = np.random.rand(3)
        done = float(i+0)/num
        point_color = np.array([0.25 + 0.75*(1-done), (2*(0.25 + 0.75*done))%1, i%2])
        line_color = .6*point_color
        scenes = getScenes('subject is "{}" and input is "{}" and output is "{}"'.format(subject, input, output))
        _=map(lambda s: partialCylinderCompress(s, start, end), scenes)
        num_tracks = len(scenes)
        track_idx = 0
        for scene in scenes:
            #point_color[2] = float(track_idx+1) / num_tracks
            #line_color = 0.6 * point_color
            track_idx += 1
            plotScene(scene, line_color=tuple(line_color), point_color=tuple(point_color), tube_sides=3)
        i += 1
    return f

            
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

