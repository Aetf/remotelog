# [SublimeLinter pylint-@python: 2]
#!/usr/bin/python2

import os, time, sys


current_milli_time = lambda: int(round(time.time() * 1000))

totals = {}
idles = {}
baseTime = 0
#map_cpus = {"cpu":-1, "cpu0":1, "cpu8":2, "cpu4":3, "cpu12":4, "cpu2":5, "cpu10":6, "cpu6":7, "cpu14":8, "cpu1":9, "cpu9":10, "cpu5":11, "cpu13":12, "cpu3":13, "cpu11":14, "cpu7":15, "cpu15":16}
map_cpus = {"cpu":-1, "cpu0":1, "cpu1":2, "cpu4":3, "cpu5":4, "cpu2":5, "cpu3":6, "cpu6":7, "cpu7":8}
def store_utilization(flag=False, filename=None):
    global baseTime
    if filename:
        fw = open(filename, 'a')
    fd = open('/proc/stat','r')
    if flag:
        fw.write('Timestamp: {}\n'.format(current_milli_time()))
    for line in fd:
        if line.find('cpu')<0:
            continue
        list1 = line.split()
        if baseTime==0:
            baseTime = long(list1[4])
        res = 0.0
        idle = long(list1[4])
        for i in range(1, len(list1)):
            res += long(list1[i])
        #print "idles time of core ", list1[0], 100.0*idle/res
        if flag==False:
            #print "idles time of core ", list1[0], 100.0*idle/res
            totals[list1[0]] = res
            idles[list1[0]] = idle
        else:
            if res==totals[list1[0]]:
                continue
            if filename:
                #fw.write(str(map_cpus[list1[0]])+"\t"+str(100.0-100.0*(idle - idles[list1[0]])/(res-totals[list1[0]]))+"\n")
                cpu = list1[0]
                fw.write('{}\t{}\n'.format(cpu,
                                           100.0-100.0*(idle - idles[cpu])/(res-totals[cpu])
                                          ))
            else:
                print list1[0], 100.0-100.0*(idle - idles[list1[0]])/(res-totals[list1[0]])

    if flag:
        fw.write('---\n'.format(current_milli_time()))
    fd.close()
    if filename:
        fw.close()

def compute_utilization(duration=1, output=None):
    store_utilization()
    time.sleep(duration)
    store_utilization(flag=True, filename=output)


if __name__=='__main__':
    output_dir = '/home/peifeng/storm-0.10.0/logs'
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    while True:
        compute_utilization(duration=1, output=output_dir + '/log.cpu')
