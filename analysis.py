"""Log Analysis"""
import re
import glob
import os
import sys
import time
from copy import copy, deepcopy
from collections import defaultdict
from itertools import groupby
from statistics import mean

import matplotlib as mpl
import pandas as pd
import seaborn as sns
from scipy.stats import f_oneway, linregress

global_debug = False
stages = ['spout', 'scale', 'fat_features', 'drawer', 'streamer', 'ack']
stages2idx = {stages[idx]: idx for idx in range(0, len(stages))}

full_stages = []
for _cur, _nxt in zip(stages[:-1], stages[1:]):
    full_stages.append(_cur)
    full_stages.append(_cur + '-' + _nxt)
full_stages.append(stages[-1])

categories = ['waiting'] + full_stages + ['finished', 'failed']
cat2idx = {categories[idx]: idx for idx in range(0, len(categories))}


def prev_stage(stage):
    """Previous stage"""
    return stages[stages2idx[stage] - 1]


def next_stage(stage):
    """Next stage"""
    return stages[stages2idx[stage] + 1]


globpattern = None
def _globpattern():
    """Return the glob pattern"""
    if not globpattern is None:
        return globpattern
    return time.strftime("archive/%Y-%-m-%d/1")


def read_log(filename, pattern):
    """Read log entries from file"""
    with open(filename) as f:
        lines = f.readlines()
    return [pattern.match(line).groupdict() for line in lines if pattern.match(line)]


def correct_log_type(logs):
    """Correct log entry data types"""
    for log in logs:
        log['req'] = int(log['req'])
        log['seq'] = int(log['seq'])
        if log['stage'] == 'queue':
            log['stage'] = 'spout'
        elif log['stage'] == 'fetcher':
            log['stage'] = 'spout'
        log['stamp'] = int(log['stamp'])
        if not log['size'] is None:
            log['size'] = int(log['size'])
        else:
            log['size'] = 0
    # Normalize time stamp
    start_time = min([l['stamp'] for l in logs])
    for l in logs:
        l['stamp'] -= start_time
    return logs


def group_by_frame(logs):
    """Group log entries by frame"""
    return group_by(logs, 'seq')


def group_by(logs, attr):
    """group list by attr"""
    tmp = defaultdict(list)
    getter = frame_key_getter(attr)
    for log in logs:
        tmp[getter(log)].append(log)
    return list(tmp.values())


def frame_key_getter(*args):
    """Return frame compare key"""
    def getter(frame):
        """Inner"""
        keys = []
        for k in args:
            if k == 'stage':
                if frame['stage'] in stages2idx:
                    keys.append(stages2idx[frame[k]])
                else:
                    keys.append(stages2idx['fat_features']+0.5)
            else:
                keys.append(frame[k])
        return tuple(keys)
    return getter


def flattened(l):
    """Flatten a list of list"""
    return [item for sublist in l for item in sublist]


def str_range(l):
    """Print a int list as 1-4,7,10,40-50"""
    ranges = []
    for _, group in groupby(enumerate(l), lambda index_item: index_item[0] - index_item[1]):
        group = [item[1] for item in group]
        #group = map(itemgetter(1), group)
        if len(group) > 1:
            ranges.append('{}-{}'.format(group[0], group[-1]))
        else:
            ranges.append('{}'.format(group[0]))
    return ','.join(ranges)


def tidy_frame_logs(logs_per_frame, debug=False):
    """Fix odd cross stage log orders"""
    res = sorted(deepcopy(logs_per_frame), key=frame_key_getter('req', 'stamp', 'stage'))
    counter = 0
    # Fix consecutive entering or leaving entries
    for idx in range(1, len(res)-1):
        if (res[idx]['evt'] == 'Entering'
                and res[idx-1]['evt'] == 'Entering'
                and res[idx-1]['stage'] == res[idx+1]['stage']
                and res[idx-1]['seq'] == res[idx]['seq']
                and res[idx]['seq'] == res[idx+1]['seq']
           ):
            #    Enter A
            # -> Enter B
            #    Leave A
            #    Leave B
            diff = res[idx+1]['stamp'] - res[idx]['stamp']
            if debug or diff > 30:
                print('WARNING: seq {}: consecutive entering on stage {} and {} with stamp '
                      'difference {}'.format(res[idx]['seq'], res[idx-1]['stage'],
                                             res[idx]['stage'], diff), file=sys.stderr)
            res[idx]['stamp'], res[idx+1]['stamp'] = res[idx+1]['stamp'], res[idx]['stamp']
            counter += 1
        elif (res[idx]['evt'] == 'Leaving'
              and res[idx+1]['evt'] == 'Leaving'
              and res[idx-1]['stage'] == res[idx]['stage']
              and res[idx-2]['stage'] == res[idx+1]['stage']
              and res[idx-1]['seq'] == res[idx]['seq']
              and res[idx]['seq'] == res[idx+1]['seq']
             ):
            #    Enter A
            #    Enter B
            # -> Leave B
            #    Leave A
            diff = res[idx+1]['stamp'] - res[idx-1]['stamp']
            if debug or diff > 30:
                print('WARNING: seq {}: consecutive leaving on stage {} and {} with stamp '
                      'difference {}'.format(res[idx]['seq'], res[idx]['stage'],
                                             res[idx+1]['stage'], diff), file=sys.stderr)
            (res[idx-1]['stamp'],
             res[idx]['stamp'],
             res[idx+1]['stamp']) = (res[idx]['stamp'],
                                     res[idx+1]['stamp'],
                                     res[idx-1]['stamp'])
            counter += 1
        elif (res[idx]['evt'] == 'Entering'
              and res[idx-1]['evt'] == 'Entering'
              and res[idx]['stage'] == 'fat_features'
              and res[idx+1]['evt'] == 'OpBegin'
              and res[idx+2]['stage'] == 'scale'
              and res[idx-1]['seq'] == res[idx]['seq']
              and res[idx]['seq'] == res[idx+2]['seq']
             ):
            #    Enter scale
            # -> Enter fat_features
            #    OpBegin A
            #    Leave scale
            diff = res[idx+2]['stamp'] - res[idx]['stamp']
            if debug or diff > 30:
                print('WARNING: seq {}: consecutive entering on stage {} and {} with stamp '
                      'difference {}'.format(res[idx]['seq'], res[idx-1]['stage'],
                                             res[idx]['stage'], diff), file=sys.stderr)
            (res[idx]['stamp'],
             res[idx+1]['stamp'],
             res[idx+2]['stamp']) = (res[idx+2]['stamp'],
                                     res[idx+2]['stamp'],
                                     res[idx]['stamp'])
            counter += 1
        elif (res[idx]['evt'] == 'Leaving'
              and res[idx-1]['evt'] == 'Entering'
              # important, we only handle cases that not handled in first clause
              and res[idx-2]['evt'] != 'Entering'
              and res[idx-1]['stage'] == res[idx+1]['stage']
              and res[idx-1]['seq'] == res[idx]['seq']
              and res[idx]['seq'] == res[idx+1]['seq']
             ):
            #    Enter A
            #    .....
            #    Enter B
            # -> Leave A
            #    Leave B
            diff = res[idx]['stamp'] - res[idx-1]['stamp']
            if debug or diff > 30:
                print('WARNING: seq {}: consecutive leaving on stage {} and {} with stamp '
                      'difference {}'.format(res[idx]['seq'], res[idx-1]['stage'],
                                             res[idx]['stage'], diff), file=sys.stderr)
            res[idx-1]['stamp'], res[idx]['stamp'] = res[idx]['stamp'], res[idx-1]['stamp']
            counter += 1
        elif (res[idx]['evt'] == 'Leaving'
              and res[idx+1]['evt'] == 'Entering'
              and res[idx]['stage'] == 'streamer'
              and res[idx+1]['stage'] == 'streamer'
              and res[idx]['seq'] == res[idx+1]['seq']
             ):
            # -> Leave streamer
            #    Enter streamer
            diff = res[idx+1]['stamp'] - res[idx]['stamp']
            if True or debug or diff > 5:
                print('WARNING: seq {}: inverted streamer Entering/Leaving'
                      ' with stamp difference {}'.format(res[idx]['seq'], diff), file=sys.stderr)
            res[idx]['stamp'], res[idx+1]['stamp'] = res[idx+1]['stamp'], res[idx]['stamp']
            counter += 1

    res = sorted(res, key=frame_key_getter('stamp', 'stage'))
    return (res, counter)


def load_cpu(filename):
    """Load cpu utilization data"""
    cpu = []
    record = None
    start = None
    pttn = re.compile(r'cpu(?P<cpu>\d*)\s(?P<val>[\d.]+)')
    with open(filename) as f:
        for line in f.readlines():
            if line.startswith('Timestamp'):
                record = {}
                stamp = int(line.split()[1])
                if start is None:
                    start = stamp
                record['timestamp'] = (stamp - start) / 1000
            elif line.startswith('---'):
                if record is None:
                    print('WARNING: malformated cpu log: "{}"'.format(line), file=sys.stderr)
                    continue
                cpu.append(record)
            else:
                m = pttn.match(line)
                if m is None or record is None:
                    print('WARNING: malformated cpu log: "{}"'.format(line), file=sys.stderr)
                    continue
                cpuid = m.groupdict()['cpu']
                val = float(m.groupdict()['val'])
                if cpuid == '':
                    record['average'] = val
                else:
                    record[int(cpuid)] = val
    return cpu


def collect_log(log_dir=None):
    """Collect log from files"""
    pttn = re.compile(r'^.+RequestID: (?P<req>[0-9-]+) StreamID: (?P<id>[^ ]+)'
                      r' SequenceNr: (?P<seq>\d+)'
                      r' (?P<evt>\w+) (?P<stage>[\w.]+):'
                      r' (?P<stamp>\d+)'
                      r'( Size: (?P<size>\d+))?$')
    logs = []
    cpus = {}
    if log_dir is None:
        log_dir = _globpattern()
    for machine in next(os.walk(log_dir))[1]:
        files = glob.glob(os.path.join(log_dir, machine, '*.log'))
        print('Collect log from', files, file=sys.stderr)
        tmp = []
        for file in files:
            tmp = tmp + read_log(file, pttn)
        for item in tmp:
            item['machine'] = machine
        logs = logs + tmp

        cpu_log = os.path.join(log_dir, machine, 'log.cpu')
        print('Collect cpu log from', cpu_log, file=sys.stderr)
        cpus[machine] = load_cpu(cpu_log)

    correct_log_type(logs)
    tidy_logs, corrected_counter = zip(*[tidy_frame_logs(per_frame, debug=global_debug)
                                         for per_frame in group_by_frame(logs)])
    print('Auto fixed cross stage timming issues for {} log entries'
          .format(sum(corrected_counter)), file=sys.stderr)
    return list(tidy_logs), cpus, logs


def extract_frames(tidy_logs):
    """Extract frames from tidy logs"""
    def extract_frame(frame_entries):
        """Form a single frame object"""
        frame = {
            'seq': frame_entries[0]['seq'],
            'retries': [],
            'failed': None
        }
        retries = group_by(frame_entries, 'req')
        retries.sort(key=lambda trial: trial[0]['req'])
        for trial in retries:
            trial.sort(key=frame_key_getter('stamp', 'stage'))
            head = []
            for idx in range(0, len(trial)):
                if trial[idx]['evt'] == 'Failed':
                    frame['failed'] = trial[idx]['stamp']
                evt = trial[idx]['evt']
                if evt == 'Retry':
                    evt = 'Leaving'
                    head.append(retries[0][0])
                trial[idx] = (evt, trial[idx]['stage'], trial[idx]['stamp'])
            trial[:0] = head
        frame['retries'] = [item for item in retries if isinstance(item[0], tuple)]
        return frame

    frames = [extract_frame(frame_entries) for frame_entries in tidy_logs]
    return frames


def check_frame(frame, debug=False):
    """Sanity check a single frame"""
    for trial in frame['retries']:
        in_stage = False
        last_stage = None
        for evt, stage_idx, _ in trial:
            if evt == 'Entering' and not in_stage:
                if last_stage is not None and next_stage(last_stage) != stage_idx:
                    if debug:
                        print('Non consecutive stage for frame', frame['seq'], file=sys.stderr)
                    # Non consecutive stage
                    return False
                last_stage = stage_idx
                in_stage = True
            elif evt == 'Leaving' and in_stage:
                if last_stage != stage_idx:
                    # Non consecutive stage entering/leaving
                    if debug:
                        print('Non consecutive stage entering/leaving for frame', frame['seq'],
                              ': last', last_stage, 'leaving', stage_idx, file=sys.stderr)
                    return False
                in_stage = False
            elif evt == 'OpBegin':
                continue
            elif evt == 'OpEnd':
                continue
            elif evt == 'Ack' or evt == 'Failed':
                continue
            else:
                # Unexpected trial event
                if debug:
                    print('Unexpected trial event for frame ', frame['seq'], ': ', evt,
                          file=sys.stderr)
                return False
    return True


def sanity_check(frames, debug=False):
    """A quick check of frame data"""
    cnt = 0
    for frame in frames:
        if not check_frame(frame, debug):
            print('WARNING: sanity check failed at frame', frame, file=sys.stderr)
            cnt = cnt + 1
    if cnt > 0:
        print(cnt, 'frames did not pass the check', file=sys.stderr)
    return cnt == 0


def sanity_filter(frames):
    """Filter the frames to those passed the sanity check"""
    lst = [frame for frame in frames if check_frame(frame, debug=global_debug)]
    print('Dropped {} frames'.format(len(frames) - len(lst)), file=sys.stderr)
    return lst


def compute_latency(frames):
    """Computue additional data on each frame"""
    avg_latency = defaultdict(list)
    anomaly_cnt = 0
    for frame in frames:
        latencies = defaultdict(list)
        for trial in frame['retries']:
            _, _, last_stamp = trial[0]
            last_op_stamp = -1
            for evt, stage_idx, stamp in trial[1:]:
                if evt == 'Entering':
                    # Cross stage latency
                    latency = stamp - last_stamp
                    key = '{}-{}'.format(prev_stage(stage_idx), stage_idx)
                    latencies[key].append(latency)
                    last_stamp = stamp
                elif evt == 'Leaving':
                    # In stage latency
                    latency = stamp - last_stamp
                    latencies[stage_idx].append(latency)
                    if stage_idx == 'streamer':
                        # The frame left the last stage, compute total latency
                        latency = stamp - trial[0][2]
                        latencies['total'].append(latency)
                    last_stamp = stamp
                elif evt == 'OpBegin':
                    last_op_stamp = stamp
                elif evt == 'OpEnd':
                    # In frameOp latency
                    latency = stamp - last_op_stamp
                    key = 'Op:{}'.format(stage_idx)
                    latencies[key].append(latency)
                    last_op_stamp = stamp
                elif evt == 'Ack':
                    # Last bolt to spout ack
                    latency = stamp - last_stamp
                    key = '{}-{}'.format(prev_stage(stage_idx), stage_idx)
                    latencies[key].append(latency)
                    last_stamp = stamp
        if not frame['failed'] is None and 'total' in latencies:
            # The frame failed but left the final stage
            anomaly_cnt = anomaly_cnt + 1

        # Average on each latency item and add to global avg_latency
        for k in latencies.keys():
            avg_latency[k] = avg_latency[k] + latencies[k]
            latencies[k] = sum(latencies[k])/len(latencies[k])
        latencies = defaultdict(int, latencies)
        latencies['service'] = (latencies['scale'] + latencies['fat_features']
                                + latencies['drawer'] + latencies['streamer'])
        frame['latencies'] = latencies

    if anomaly_cnt > 0:
        print(anomaly_cnt, 'frames left the final stage but still marked failed', file=sys.stderr)
    # Final average on global avg_latency
    for k in avg_latency.keys():
        avg_latency[k] = sum(avg_latency[k])/len(avg_latency[k])
    return avg_latency


def latency_check(frames):
    """A quick check on cross frame latency"""
    cnt = 0
    max_abs = 0
    for frame in frames:
        if 'latencies' not in frame:
            print('ERROR: run compute_latency before latency_check!!!', file=sys.stderr)
            return True
        latencies = frame['latencies']
        for k, latency in latencies.items():
            if latency < 0:
                print("WARNING: Negative latency detected for {} for seq {}!"
                      .format(k, frame['seq']), file=sys.stderr)
                max_abs = max(-latency, max_abs)
                cnt = cnt + 1
    if cnt > 0:
        print(cnt, 'instance did not pass the check', file=sys.stderr)
        print('Max negative latency abs value', max_abs, file=sys.stderr)
    return cnt == 0


def normalizeDict(d, excludeKeys=None):
    """Divide each entry by sum"""
    if excludeKeys is None:
        excludeKeys = []
    total = 0
    keys = [key for key in d.keys() if key not in excludeKeys]
    for k in keys:
        total = total + d[k]
    for k in keys:
        if total != 0:
            d[k] = d[k] / total
        else:
            d[k] = 1 / len(keys)
    return d


def show_log(logs, seq):
    """Print log entries for specific frame"""
    single = [log for log in logs if log['seq'] == seq]
    single = sorted(single, key=frame_key_getter('stamp'))
    for item in single:
        print('Machine: {:9} Seq: {:<5}\t{:<8} {:^12}: {}\tSize: {}'
              .format(item['machine'], item['seq'], item['evt'], item['stage'],
                      item['stamp'], item['size']))


def compute_fps(tidy_logs, stage='spout', evt='Entering', step=1000):
    """Compute fps on spout"""
    # Flatten logs for processing according to stamp
    flaten_logs = flattened(tidy_logs)
    flaten_logs.sort(key=frame_key_getter('stamp', 'stage'))

    start_time = flaten_logs[0]['stamp']
    curr_bin = -1
    curr_cnt = 0
    fps = []
    for log in flaten_logs:
        t = int((log['stamp'] - start_time) / step)
        if t != curr_bin:
            # Moving to next step, commit current fps
            curr_bin = t
            fps.append(curr_cnt / step * 1000)
            curr_cnt = 0
        if log['evt'] == evt and log['stage'] == stage:
            curr_cnt = curr_cnt + 1
    fps.append(curr_cnt / step * 1000)
    return fps


def compute_stage_dist(tidy_logs, normalize=True, step=1000, excludeCat=None):
    """Compute stage distribution per step"""
    # Flatten logs for processing according to stamp
    flaten_logs = [item for sublist in tidy_logs for item in sublist]
    flaten_logs.sort(key=frame_key_getter('stamp', 'stage'))

    # Variables that must compute first
    frame_cnt = len(tidy_logs)
    start_time = flaten_logs[0]['stamp']
    # Prepare an initial distribute
    dist = {cat: 0 for cat in categories}
    dist['waiting'] = frame_cnt
    dist['time'] = 0

    distributions = []
    last_stage = defaultdict(int)
    curr_time = -1
    for log in flaten_logs:
        t = int((log['stamp'] - start_time) / step)
        if t != curr_time:
            # moving to next step, commit current dist
            curr_time = t
            for v in dist.values():
                if v < 0:
                    print('WARNING: negative size', dist, file=sys.stderr)
            tmp = copy(dist)
            if normalize:
                if excludeCat is None:
                    excludeCat = []
                else:
                    excludeCat = copy(excludeCat)
                excludeCat.append('time')
                normalizeDict(tmp, excludeKeys=excludeCat)
            distributions.append(tmp)
            dist['time'] = t
        seq = log['seq']
        fr = last_stage[seq]
        if log['evt'] == 'Entering':
            to = cat2idx[log['stage']]
        elif log['evt'] == 'Leaving':
            to = cat2idx[log['stage']] + 1
        elif log['evt'] == 'Retry':
            fr = -1
            to = cat2idx[log['stage']] + 1
        elif log['evt'] == 'Failed':
            #to = cat2idx['failed']
            pass
        else:
            #print('Unknown event at log entry', log)
            continue
        if fr != -1:
            dist[categories[fr]] = dist[categories[fr]] - 1
        dist[categories[to]] = dist[categories[to]] + 1
        last_stage[seq] = to

    distributions.sort(key=frame_key_getter('time'))
    return distributions


def fps_plot(tidy_logs, point=('Entering', 'spout'), step=1000, **kwargs):
    """Plot!"""
    if not isinstance(point, list):
        point = [point]

    fpses = pd.DataFrame({
        '{} {}'.format(evt, stage): compute_fps(tidy_logs, stage=stage, evt=evt, step=step)
        for evt, stage in point
    })
    fpses.loc[:, 'time'] = [i * step / 1000 for i in range(0, len(fpses.index))]
    thePlot = fpses.plot(x='time', **kwargs)
    thePlot.set_ylabel('Frame per second')
    thePlot.set_xlabel('Time (s)')
    thePlot.figure.tight_layout()
    return thePlot


def cpu_plot(cpu, which=None):
    """Plot!"""
    if which is None:
        which = 'average'
    fig, axes = sns.plt.subplots(nrows=len(cpu.keys()), ncols=1)
    if len(cpu.keys()) == 1:
        axes = [axes]
    for idx, m in enumerate(cpu.keys()):
        df = pd.DataFrame.from_dict(cpu[m])
        thePlot = df.plot(x='timestamp', y=which, ax=axes[idx])
        thePlot.set_title(m)
        thePlot.set_xlabel('Time (s)')
        thePlot.set_ylabel('CPU utilization (%)')
        thePlot.figure.tight_layout()
    return fig


def cdf_plot(clean_frames, **kwargs):
    """Plot!"""
    ser = pd.Series([frame['latencies']['total'] for frame in clean_frames])
    thePlot = ser.hist(cumulative=True, histtype='step', bins=2000, **kwargs)
    thePlot.set_xlabel('Latency (ms)')
    thePlot.set_ylabel('Frame count')
    thePlot.figure.tight_layout()
    return thePlot


def time_latency_plot(clean_frames, stage='total', **kwargs):
    """Plot!"""
    if not isinstance(stage, list):
        stage = [stage]

    ff = sorted(clean_frames, key=frame_key_getter('seq'))
    ff = [frame for frame in ff
          if set.issubset(set(stage), frame['latencies'].keys())]
    df = pd.DataFrame({
        s: pd.Series([frame['latencies'][s] for frame in ff])
        for s in stage
    })
    df.loc[:, 'SequenceNr'] = [frame['seq'] for frame in ff]
    thePlot = df.plot(x='SequenceNr', **kwargs)
    thePlot.set_xlabel('SequenceNr')
    thePlot.set_ylabel('Latency (ms)')
    thePlot.get_xaxis().set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    thePlot.figure.tight_layout()
    return thePlot


def latency_plot(clean_frames, latencies=None, **kwargs):
    """Plot!"""
    if latencies is None:
        latencies = full_stages[:-2] + ['service', 'total']
    latenciesDf = pd.DataFrame.from_dict([frame['latencies'] for frame in clean_frames])
    sns.plt.figure()
    latAx = sns.barplot(data=latenciesDf, order=latencies, saturation=0.4, **kwargs)
    latAx.set_yscale('log')
    latAx.set_ylabel('Latency (ms)')
    latAx.set_xticklabels(latAx.get_xticklabels(), rotation=30, size='x-small')
    #latAx.tick_params(axis='x', which='major', labelsize=6)
    #latAx.tick_params(axis='x', which='minor', labelsize=4)
    for p in latAx.patches:
        value = '{:.2f}'.format(p.get_height())
        pos = (p.get_x() + p.get_width()/2,
               p.get_height() if p.get_height() > 0 else 10)
        latAx.annotate(value, xy=pos,
                       xytext=(0, 8), xycoords='data', textcoords='offset points',
                       size='medium', ha='center', va='center')
    latAx.figure.tight_layout()
    return latAx


def distribution_plot(tidy_logs, normalize=True, step=200, excludeCat=None, **kwargs):
    """Plot!"""
    dists = compute_stage_dist(tidy_logs, normalize, step, excludeCat)
    distDf = pd.DataFrame.from_dict(dists)
    cols = copy(categories)
    cols.reverse()
    cols = ['time'] + cols
    distDf = distDf[cols]

    if not excludeCat is None:
        for cat in excludeCat:
            del distDf[cat]
    distAx = distDf.plot.area(x='time', **kwargs)
    distAx.set_xlabel('Time (x {}ms)'.format(step))
    distAx.set_ylabel('Ratio')
    distAx.set_ylim([0, 1])
    distAx.figure.tight_layout()
    return distAx


def start_dist_of_failed(clean_frames, **kwargs):
    """Plot!"""
    failed = [(int(f['failed']/1000), int(f['retries'][0][0][2]/1000))
              for f in clean_frames if f['failed'] is not None]

    failed_at = []
    for t in range(0, max(failed)[0]):
        l = len([f for f, _ in failed if f >= t and f < (t+1)])
        if l > 0:
            print('{}: {}'.format(t, l))
            failed_at.append(t)

    d = [[s for f, s in failed if f == at] for at in failed_at]
    fd = flattened(d)
    minimum, maximum = min(fd), max(fd)
    sns.plt.hist(d, bins=[i - 0.5 for i in range(minimum, maximum + 1)], stacked=True,
                 label=['failed at {}'.format(at) for at in failed_at], **kwargs)
    thePlot = sns.plt.gca()
    thePlot.legend()
    return thePlot

def run(log_dir=None):
    """Read log and parse"""
    #pttn = re.compile(r'^.+SequenceNr: (?P<seq>\d+)' +
    #                  r' (?P<evt>\w+) (?P<stage>\w+):' +
    #                  r' (?P<stamp>\d+)' +
    #                  r'( \(latency (?P<latency>\d+)\))?$')
    tidy_logs, cpus, raw_logs = collect_log(log_dir)
    tidy_logs.sort(key=lambda per_frame_logs: per_frame_logs[0]['seq'])

    frames = extract_frames(tidy_logs)

    clean_frames = sanity_filter(frames)

    compute_latency(clean_frames)

    latency_check(clean_frames)

    distributions = compute_stage_dist(tidy_logs)

    mpl.rcParams['figure.dpi'] = 193
    mpl.rcParams['axes.formatter.useoffset'] = False
    sns.plt.ion()
    sns.set_style('whitegrid')

    return tidy_logs, frames, clean_frames, distributions, cpus, raw_logs

class exp_res(object):
    """An object holds all parsed logs for one experiment"""
    def __init__(self, exp_name):
        self.exp_name = exp_name
        log_dir = 'archive/{}'.format(exp_name)
        (self.tidy_logs, self.frames, self.clean_frames,
         self.distributions, self.cpus, self.raw_logs) = run(log_dir)
        self.seqs = sorted([f['seq'] for f in self.clean_frames])
        self._load_params(os.path.join(log_dir, 'params.txt'))

    def __str__(self):
        """Print"""
        return '<exp {} nodes:{}, FPS:{}, topo: {}>'.format(self.exp_name,
                                                            len(self.params['nodes']),
                                                            self.params['fps'],
                                                            self._topology())

    def __repr__(self):
        return self.__str__()

    def _topology(self):
        """A string repr of the topology"""
        return '{scale}+{fat}+{drawer}'.format(**self.params)

    def _load_params(self, params_path):
        """Load parameters"""
        self.params = {}
        with open(params_path) as f:
            lines = [(line[:-1] if line.endswith('\n') else line) for line in f.readlines()]
            for line in lines:
                if line.startswith('--'):
                    key, value = line[2:].split('=')
                    self.params[key] = value
                elif '=' in line:
                    self.params['cpu_per_node'] = line.split('=')[1]
                else:
                    self.params['topology_class'] = line
        # node count
        nodes = set()
        for log in flattened(self.tidy_logs):
            nodes.add(log['machine'])
        self.params['nodes'] = nodes
        # parse numbers
        for key in ['fps', 'scale', 'fat', 'drawer']:
            self.params[key] = int(self.params[key])

    def _select(self, seq, raw=False):
        """Select a subset of frames using seq"""
        if seq is None:
            seq = self.seqs
        try:
            seq = set(seq)
        except TypeError:
            seq = set([seq])

        ff = self.frames if raw else self.clean_frames

        selected = [frame for frame in ff
                    if frame['seq'] in seq]
        return selected, seq

    def show_log(self, seq=None):
        """Print raw log entries"""
        if seq is not None and not isinstance(seq, list):
            seq = [seq]
        elif seq is None:
            seq = self.seqs

        for s in seq:
            show_log(self.tidy_logs[s], s)

    def show_frame(self, seq=None, raw=True):
        """Print frame entries"""
        for frame in self._select(seq, raw)[0]:
            print('Seq: {:<5}\tRetries: {:<2}\tFailed: {}'
                  .format(frame['seq'], len(frame['retries']), frame['failed']))
            for trial in frame['retries']:
                print('------------------')
                for evt, stage_idx, stamp in trial:
                    print('{:<8} {:^12}: {}'.format(evt, stage_idx, stamp))

    def param_core(self):
        """Get cores"""
        return int(self.params['scale']) + int(self.params['fat']) + int(self.params['drawer']) + 3

    def latency(self, seq=None, title=None, **kwargs):
        """Print and plot selected frames. seq can be a list or a single number"""
        frames, frame_seqs = self._select(seq)
        p = latency_plot(frames, **kwargs)
        if title is None:
            p.set_title('Frame {}'.format(str_range(frame_seqs) if seq is not None else 'All'))
        else:
            p.set_title(title)
        p.figure.canvas.set_window_title('Exp: {}'.format(self.exp_name))
        p.figure.tight_layout()
        return p

    def seq_latency(self, stage='total', seq=None, **kwargs):
        """Plot latency to seq"""
        p = time_latency_plot(self._select(seq)[0], stage, **kwargs)
        p.figure.canvas.set_window_title('Exp: {}'.format(self.exp_name))
        p.figure.tight_layout()
        return p

    def fps(self, point=None, step=1000, **kwargs):
        """Plot FPS at stage"""
        if point is None:
            point = [('Entering', 'spout'), ('Ack', 'ack')]
        p = fps_plot(self.tidy_logs, point=point, step=step, **kwargs)
        p.figure.canvas.set_window_title('Exp: {}'.format(self.exp_name))
        p.figure.tight_layout()
        return p

    def cpu(self, which=None, **kwargs):
        """Plot CPU usage"""
        p = cpu_plot(self.cpus, which, **kwargs)
        p.canvas.set_window_title('Exp: {}'.format(self.exp_name))
        p.tight_layout()
        return p

    def avg_fps(self, stage='spout', evt='Entering', trim=False, limit=None):
        """Calculate average fps"""
        fpses = compute_fps(self.tidy_logs, stage, evt)
        if limit is not None:
            fpses = fpses[:limit]
        while trim and fpses[-1] == 0.0:
            fpses.pop()
        return (mean(fpses), len(fpses))

    def avg_latency(self, which=None, useLinregress=True, detail=False):
        """Calculate average latency"""
        if which is None:
            which = 'total'
        def avg1():
            """Use linregress"""
            # cut begining and ending
            begin_at = 0.25 if len(self.seqs) > 500 else 0.05
            end_at = 0.75 if len(self.seqs) > 500 else 0.95

            begin_at = int(len(self.seqs) * begin_at)
            end_at = int(len(self.seqs) * end_at)
            samples = [(f['seq'], f['latencies'][which])
                       for f in self._select(self[0].seqs[begin_at:end_at])
                       if which in f['latencies']]

            slope, intercept, _, pvalue, sd = linregress(samples)
            if detail or (pvalue <= 0.05 and slope > 0.1):
                # they are not same
                print('WARNING: {}: possibly bad data, latency values aren\'t stable.'
                      ' slope={:.3f}, intercept={:.2f}, pvalue={:.2e}, stderr={:.4f},'
                      ' sample_size={}'.format(self, slope, intercept, pvalue, sd, len(samples)),
                      file=sys.stderr)
            return mean([lat for _, lat in samples])
        def avg2():
            """Use foneway"""
            # sample only from the mid part
            sample_ratio = 0.25 if len(self.seqs) > 500 else 0.05
            st_idx = int(len(self.seqs) * sample_ratio)
            mid_idx = int(len(self.seqs) * sample_ratio * 2)
            ed_idx = int(len(self.seqs) * sample_ratio * 3)
            samples = []
            samples.append(self._select(self[0].seqs[st_idx:mid_idx]))
            samples.append(self._select(self[0].seqs[mid_idx:ed_idx]))
            # sampled latencies
            lats = [[f['latencies'][which] for f in sample if which in f['latencies']]
                    for sample in samples]
            # test if they are same
            _, pvalue = f_oneway(*lats)
            if detail or pvalue <= 0.05:
                # they are not same
                print('WARNING: {}: possibly bad data, latency values aren\'t stable.'
                      ' pvalue={:.4e}, sample_size={}, avg={}'
                      .format(self, pvalue, [len(l) for l in lats], [mean(l) for l in lats]),
                      file=sys.stderr)

            return mean(flattened(lats))

        return avg1() if useLinregress else avg2()


class cross_res(object):
    """Cross analysis of multiple runs of experiments"""
    def __init__(self, *args):
        self.exps = [exp_res(arg) for arg in args]

    def latency(self, which='total', x=None, **kwargs):
        """Average latency"""
        if not isinstance(which, list):
            which = [which]
        df = pd.DataFrame({
            k: pd.Series([exp.avg_latency(k) for exp in self.exps])
            for k in which
        })

        if x is None:
            x = ('Threads for fat_features', lambda exp: int(exp.params['fat']))
        x_name, x_data = x
        df.loc[:, x_name] = [x_data(exp) for exp in self.exps]
        df = df.sort_values(x_name)
        p = df.plot(x=x_name, marker='o', **kwargs)
        p.set_ylabel('Latency (ms)')
        return p, df

    def fps(self, points=None, x=None, **kwargs):
        """Average fps"""
        if points is None:
            points = [('Entering', 'spout'), ('Ack', 'ack')]
        df = pd.DataFrame({
            '{} {}'.format(evt, stage): pd.Series([exp.avg_fps(stage, evt, trim=True)[0]
                                                   for exp in self.exps])
            for evt, stage in points
        })

        if x is None:
            x = ('Threads for fat_features', lambda exp: int(exp.params['fat']))
        x_name, x_data = x

        df.loc[:, x_name] = [x_data(exp) for exp in self.exps]
        df = df.sort_values(x_name)
        p = df.plot(x=x_name, marker='o', **kwargs)
        return p
