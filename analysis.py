"""Log Analysis"""
import re
import glob
import os
import time
from copy import copy, deepcopy
from collections import defaultdict
from operator import itemgetter

import matplotlib as mpl
import pandas as pd
import seaborn as sns

stages = ['spout', 'scale', 'fat_features', 'drawer', 'streamer']
stage2idx = {stages[idx]: idx for idx in range(0, len(stages))}

full_stages = []
for _cur, _nxt in zip(stages[:-1], stages[1:]):
    full_stages.append(_cur)
    full_stages.append(_cur + '-' + _nxt)
full_stages.append(stages[-1])

categories = ['waiting'] + full_stages + ['finished', 'failed']
cat2idx = {categories[idx]: idx for idx in range(0, len(categories))}

globpattern = None
def _globpattern():
    """Return the glob pattern"""
    if not globpattern is None:
        return globpattern
    return time.strftime("archive/%Y-%-m-%d/")


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
        log['stage'] = stage2idx[log['stage']]
        log['stamp'] = int(log['stamp'])
        if not log['size'] is None:
            log['size'] = int(log['size'])
        else:
            log['size'] = 0
    return logs


def group_by_frame(logs):
    """Group log entries by frame"""
    return group_by(logs, 'seq')


def group_by(logs, attr):
    """group list by attr"""
    tmp = defaultdict(list)
    getter = itemgetter(attr)
    for log in logs:
        tmp[getter(log)].append(log)
    return list(tmp.values())


def tidy_frame_logs(logs_per_frame, debug=False):
    """Filter out old entry before retry"""
    res = sorted(deepcopy(logs_per_frame), key=itemgetter('req', 'stamp', 'stage'))
    # Fix consecutive entering or leaving entries
    for idx in range(1, len(res)-1):
        if (res[idx]['evt'] == res[idx-1]['evt']
                and res[idx]['evt'] == 'Entering'
                and res[idx-1]['stage'] == res[idx+1]['stage']):
            diff = res[idx+1]['stamp'] - res[idx]['stamp']
            if debug or diff > 30:
                print('WARNING: seq {}: consecutive entering on stage {} and {} with stamp '
                      'difference {}'.format(res[idx]['seq'], stages[res[idx-1]['stage']],
                                             stages[res[idx]['stage']],
                                             res[idx+1]['stamp'] - res[idx]['stamp']))
            res[idx]['stamp'], res[idx+1]['stamp'] = res[idx+1]['stamp'], res[idx]['stamp']
        elif (res[idx]['evt'] == res[idx+1]['evt']
              and res[idx]['evt'] == 'Leaving'
              and res[idx-1]['stage'] == res[idx]['stage']):
            diff = res[idx+1]['stamp'] - res[idx-1]['stamp']
            if debug or diff > 30:
                print('WARNING: seq {}: consecutive leaving on stage {} and {} with stamp '
                      'difference {}'.format(res[idx]['seq'], stages[res[idx]['stage']],
                                             stages[res[idx+1]['stage']],
                                             res[idx+1]['stamp'] - res[idx-1]['stamp']))
            (res[idx-1]['stamp'],
             res[idx]['stamp'],
             res[idx+1]['stamp']) = (res[idx+1]['stamp'],
                                     res[idx-1]['stamp'],
                                     res[idx]['stamp'])

    res = sorted(res, key=itemgetter('stamp', 'stage'))
    return res


def collect_log(log_dir=None):
    """Collect log from files"""
    pttn = re.compile(r'^.+RequestID: (?P<req>[0-9-]+) StreamID: (?P<id>[\w.]+)'
                      r' SequenceNr: (?P<seq>\d+)'
                      r' (?P<evt>\w+) (?P<stage>\w+):'
                      r' (?P<stamp>\d+)'
                      r'( Size: (?P<size>\d+))?$')
    logs = []
    if log_dir is None:
        log_dir = _globpattern()
    for machine in next(os.walk(log_dir))[1]:
        files = glob.glob(os.path.join(log_dir, machine, '*.log'))
        print('Collect log from', files)
        tmp = []
        for file in files:
            tmp = tmp + read_log(file, pttn)
        for item in tmp:
            item['machine'] = machine
        logs = logs + tmp

    correct_log_type(logs)
    tidy_logs = [tidy_frame_logs(per_frame) for per_frame in group_by_frame(logs)]
    return tidy_logs


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
            trial.sort(key=itemgetter('stamp', 'stage'))
            head = []
            for idx in range(0, len(trial)):
                if trial[idx]['evt'] == 'Failed':
                    frame['failed'] = trial[idx]['stamp']
                    continue
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
        last_stage = -1
        for evt, stage_idx, _ in trial:
            if evt == 'Entering' and not in_stage:
                if last_stage + 1 != stage_idx:
                    if debug:
                        print('Non consecutive stage')
                    # Non consecutive stage
                    return False
                last_stage = stage_idx
                in_stage = True
            elif evt == 'Leaving' and in_stage:
                if last_stage != stage_idx:
                    # Non consecutive stage entering/leaving
                    if debug:
                        print('Non consecutive stage entering/leaving: last',
                              stages[last_stage], 'leaving', stages[stage_idx])
                    return False
                in_stage = False
            else:
                # Unexpected trial event
                if debug:
                    print('Unexpected trial event')
                return False
    return True


def sanity_check(frames):
    """A quick check of frame data"""
    cnt = 0
    for frame in frames:
        if not check_frame(frame):
            print('WARNING: sanity check failed at frame', frame)
            cnt = cnt + 1
    if cnt > 0:
        print(cnt, 'frames did not pass the check')
    return cnt == 0


def sanity_filter(frames):
    """Filter the frames to those passed the sanity check"""
    lst = [frame for frame in frames if check_frame(frame)]
    print('Dropped {} frames'.format(len(frames) - len(lst)))
    return lst


def compute_latency(frames):
    """Computue additional data on each frame"""
    avg_latency = defaultdict(list)
    anomaly_cnt = 0
    for frame in frames:
        latencies = defaultdict(list)
        for trial in frame['retries']:
            _, _, last_stamp = trial[0]
            for evt, stage_idx, stamp in trial[1:]:
                if evt == 'Entering':
                    # Cross stage latency
                    latency = stamp - last_stamp
                    key = '{}-{}'.format(stages[stage_idx - 1], stages[stage_idx])
                    latencies[key].append(latency)
                elif evt == 'Leaving':
                    # In stage latency
                    latency = stamp - last_stamp
                    latencies[stages[stage_idx]].append(latency)
                    if stage_idx == stage2idx['streamer']:
                        # The frame left the last stage, compute total latency
                        latency = stamp - trial[0][2]
                        latencies['total'].append(latency)
                else:
                    print('WARNING: Someting wrong with frame', frame, 'Try run sanity check first')
                last_stamp = stamp
        if not frame['failed'] is None and 'total' in latencies:
            # The frame failed but left the final stage
            anomaly_cnt = anomaly_cnt + 1

        # Average on each latency item and add to global avg_latency
        for k in latencies.keys():
            avg_latency[k] = avg_latency[k] + latencies[k]
            latencies[k] = sum(latencies[k])/len(latencies[k])
        frame['latencies'] = latencies

    if anomaly_cnt > 0:
        print(anomaly_cnt, 'frames left the final stage but still marked failed')
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
            print('ERROR: run compute_latency before latency_check!!!')
            return True
        latencies = frame['latencies']
        for k, latency in latencies.items():
            if latency < 0:
                print("WARNING: Negative latency detected for {} for seq {}!"
                      .format(k, frame['seq']))
                max_abs = max(-latency, max_abs)
                cnt = cnt + 1
    if cnt > 0:
        print(cnt, 'instance did not pass the check')
        print('Max negative latency abs value', max_abs)
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


def compute_fps(tidy_logs, stage='spout', evt='Entering'):
    """Compute fps on spout"""
    # Flatten logs for processing according to stamp
    flaten_logs = [item for sublist in tidy_logs for item in sublist]
    flaten_logs.sort(key=itemgetter('stamp', 'stage'))

    start_time = flaten_logs[0]['stamp']
    curr_time = -1
    curr_cnt = 0
    fps = []
    for log in flaten_logs:
        if log['evt'] != evt or log['stage'] != stage2idx[stage]:
            continue
        t = int((log['stamp'] - start_time) / 1000)
        if t != curr_time:
            # Moving to next step, commit current fps
            curr_time = t
            fps.append(curr_cnt)
            curr_cnt = 0
        curr_cnt = curr_cnt + 1
    return fps


def compute_stage_dist(tidy_logs, normalize=True, step=1000, excludeCat=None):
    """Compute stage distribution per step"""
    # Flatten logs for processing according to stamp
    flaten_logs = [item for sublist in tidy_logs for item in sublist]
    flaten_logs.sort(key=itemgetter('stamp', 'stage'))

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
                    print('WARNING: negative size', dist)
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
        cat_idx = cat2idx[stages[log['stage']]]
        if log['evt'] == 'Entering':
            to = cat_idx
        elif log['evt'] == 'Leaving':
            to = cat_idx + 1
        elif log['evt'] == 'Retry':
            fr = -1
            to = cat_idx + 1
        elif log['evt'] == 'Failed':
            #to = cat2idx['failed']
            pass
        else:
            print('Unknown event at log entry', log)
        if fr != -1:
            dist[categories[fr]] = dist[categories[fr]] - 1
        dist[categories[to]] = dist[categories[to]] + 1
        last_stage[seq] = to

    distributions.sort(key=itemgetter('time'))
    return distributions


def print_frame(frames, seq=None):
    """Print selected frames or single frame"""
    if not seq is None and not isinstance(seq, list):
        seq = [seq]
    def show(frame):
        """Print a single frame"""
        print('Seq: {:<5}\tRetries: {:<2}\tFailed: {}'
              .format(frame['seq'], len(frame['retries']), frame['failed']))
        for trial in frame['retries']:
            print('------------------')
            for evt, stage_idx, stamp in trial:
                print('{:<8} {:^12}: {}'.format(evt, stages[stage_idx], stamp))

    if isinstance(frames, list):
        for frame in frames:
            if seq is None or frame['seq'] in seq:
                show(frame)
    else:
        show(frames)


def show_frame(logs, seq):
    """Print log entries for specific frame"""
    single = [log for log in logs if log['seq'] == seq]
    single = sorted(single, key=itemgetter('stamp'))
    for item in single:
        print('Machine: {:9} Seq: {:<5}\t{:<8} {:^12}: {}\tSize: {}'
              .format(item['machine'], item['seq'], item['evt'], stages[item['stage']],
                      item['stamp'], item['size']))


def latency_plot(clean_frames):
    """Plot!"""
    latenciesDf = pd.DataFrame.from_dict([frame['latencies'] for frame in clean_frames])
    latAx = sns.barplot(data=latenciesDf, order=(full_stages + ['total']), saturation=0.4)
    latAx.set_yscale('log')
    latAx.set_ylabel('latency (ms)')
    latAx.set_xticklabels(latAx.get_xticklabels(), rotation=30)
    for p in latAx.patches:
        value = '{:.2f}'.format(p.get_height())
        latAx.annotate(value, xy=(p.get_x() + p.get_width()/2, p.get_height()),
                       xytext=(0, 8), xycoords='data', textcoords='offset points',
                       size='small', ha='center', va='center')
    latAx.figure.tight_layout()
    return latAx


def distribution_plot(tidy_logs, normalize=True, step=200, excludeCat=None):
    """Plot!"""
    dists = compute_stage_dist(tidy_logs, normalize, step, excludeCat)
    distDf = pd.DataFrame.from_dict(dists)
    cols = copy(categories)
    cols.reverse()
    cols = ['time'] + cols
    distDf = distDf[cols]

    if not excludeCat is None:
        for cat in excludeCat:
            print(cat)
            del distDf[cat]
    distAx = distDf.plot.area(x='time')
    distAx.set_xlabel('Time (x {}ms)'.format(step))
    distAx.set_ylabel('Ratio')
    distAx.set_ylim([0,1])
    distAx.figure.tight_layout()
    return distAx

def run():
    """Read log and parse"""
    #pttn = re.compile(r'^.+SequenceNr: (?P<seq>\d+)' +
    #                  r' (?P<evt>\w+) (?P<stage>\w+):' +
    #                  r' (?P<stamp>\d+)' +
    #                  r'( \(latency (?P<latency>\d+)\))?$')
    tidy_logs = collect_log()

    frames = extract_frames(tidy_logs)

    clean_frames = sanity_filter(frames)

    compute_latency(clean_frames)

    latency_check(clean_frames)

    distributions = compute_stage_dist(tidy_logs)

    mpl.rcParams['figure.dpi'] = 193
    sns.plt.ion()
    sns.set_style('whitegrid')

    return tidy_logs, frames, clean_frames, distributions
