"""Log Analysis"""
import re
import glob
import os
import sys
import time
from datetime import datetime as dt
from copy import copy, deepcopy
from collections import defaultdict
from itertools import groupby
from statistics import mean

import numpy as np
import matplotlib as mpl
import pandas as pd
import seaborn as sns
from scipy.stats import f_oneway, linregress

global_debug = False
global_perBatch = False
global_skipAutoFix = False
global_skipOp = True

topology_stage_map = {
    'nl.tno.stormcv.deploy.DNNTopology':
        ['spout', 'scale', 'fat_features', 'drawer', 'streamer_batcher', 'streamer', 'ack'],
    'nl.tno.stormcv.deploy.BatchDNNTopology':
        ['spout', 'scale', 'dnn_forward_batcher', 'dnn_forward',
         'drawer', 'streamer_batcher', 'streamer', 'ack'],
    'nl.tno.stormcv.deploy.SpoutOnly':
        ['spout', 'scale', 'noop', 'ack'],
    'nl.tno.stormcv.deploy.SplitDNNTopology':
        ['spout', 'scale', 'face_detect', 'dnn_forward', 'dnn_classify',
         'drawer', 'streamer_batcher', 'streamer', 'ack'],
    'nl.tno.stormcv.deploy.ObjTrackingTopology':
        ['spout', 'scale', 'obj_track_batcher', 'obj_track',
         'drawer', 'streamer_batcher', 'streamer', 'ack'],
    'nl.tno.stormcv.deploy.LoopTopology':
        ['spout', 'scale', 'obj_track_batcher', 'obj_track',
         'drawer', 'streamer_batcher', 'streamer', 'ack'],
    'nl.tno.stormcv.deploy.E4_SequentialFeaturesTopology':
        ['spout', 'scale', 'fat_features', 'drawer', 'streamer_batcher', 'streamer', 'ack'],
    'nl.tno.stormcv.deploy.E3_MultipleFeaturesTopology':
        ['spout', 'scale', 'face', 'sift', 'combiner',
         'drawer', 'streamer_batcher', 'streamer', 'ack'],
    'nl.tno.stormcv.deploy.CaptionerTopology':
        ['spout', 'scale', 'vgg_feature', 'frame_grouper_batcher', 'frame_grouper',
         'captioner', 'streamer', 'ack'],
    'nl.tno.stormcv.deploy.E4_SequentialFeaturesTopology':
        ['spout', 'scale', 'fat_features','drawer', 'streamer_batcher', 'streamer', 'ack'],
}
stages = []
stages2idx = {}
full_stages = []
categories = []
cat2idx = {}

convert_batch = False
storm102 = True
skipOp = True
has_gpu_log = False


def last_stage():
    """Last stage"""
    if stages[-1] == 'ack':
        return stages[-2]
    return stages[-1]


def prev_stage(stage):
    """Previous stage"""
    return stages[stages2idx[stage] - 1]


def next_stage(stage):
    """Next stage"""
    return stages[stages2idx[stage] + 1]


def is_after(stageA, stageB):
    """If stageA comes after stageB"""
    return stages2idx[stageA] > stages2idx[stageB]


def is_before(stageA, stageB):
    """If stageA comes before stageB"""
    return stages2idx[stageA] < stages2idx[stageB]


def frame_time_for(frame, evt, stage, trial=0):
    """Return time for evt, stage"""
    trials = frame['retries']

    for e, s, stamp in trials[trial]:
        if e == evt and s == stage:
            return stamp
    return -1


def range_filter(seq_range):
    """Return a frame_filter func that selects frames/logs in seq_range"""
    def ff(tidy_logs, frames, clean_frames):
        t = [per_frame_logs for per_frame_logs in tidy_logs if per_frame_logs[0]['seq'] in seq_range]
        f = [frame for frame in frames if frame['seq'] in seq_range]
        cf = [frame for frame in clean_frames if frame['seq'] in seq_range]
        return t, f, cf
    return ff


def update_stage_info(topology_class, log_version=1):
    """Recompute stage info"""
    global stages, stages2idx, full_stages, categories, cat2idx, skipOp, storm102, convert_batch, has_gpu_log
    new_stages = topology_stage_map[topology_class]
    stages = copy(new_stages)

    # initial flag values
    convert_batch = False
    storm102 = False
    skipOp = True
    has_gpu_log = False

    if log_version < 2:
        print('INFO: log version < 2 (before 2016-7-18), remove batcher stage',
              file=sys.stderr)
        for s in new_stages:
            if s.endswith('batcher'):
                stages.remove(s)
    if log_version >= 3:
        print('INFO: log version >= 3 (after 2016-9-1), use storm102 log structure',
              file=sys.stderr)
        storm102 = True
    if log_version >= 4:
        print('INFO: log version >= 4 (after 2016-9-14), remove scale stage from topologies',
              file=sys.stderr)
        if topology_class in ['nl.tno.stormcv.deploy.DNNTopology',
                              'nl.tno.stormcv.deploy.BatchDNNTopology',
                              'nl.tno.stormcv.deploy.SpoutOnly',
                              'nl.tno.stormcv.deploy.SplitDNNTopology',
                              'nl.tno.stormcv.deploy.ObjTrackingTopology',
                              'nl.tno.stormcv.deploy.CaptionerTopology']:
            stages.remove('scale')
    if log_version >= 5:
        print('INFO: log version >= 5 (after 2016-9-25), enable gpu log', file=sys.stderr)
        has_gpu_log = True

    stages2idx = {stages[idx]: idx for idx in range(0, len(stages))}

    full_stages = []
    for _cur, _nxt in zip(stages[:-1], stages[1:]):
        full_stages.append(_cur)
        full_stages.append(_cur + '-' + _nxt)
    full_stages.append(stages[-1])
    for stage in stages:
        if stage.endswith('batcher'):
            s = stage + '-' + next_stage(stage)
            if s in full_stages:
                full_stages.remove(s)

    categories = ['waiting'] + full_stages + ['finished', 'failed']
    cat2idx = {categories[idx]: idx for idx in range(0, len(categories))}

    if topology_class == 'nl.tno.stormcv.deploy.DNNTopology' and not global_skipOp:
        skipOp = False
    else:
        skipOp = True

    if topology_class == 'nl.tno.stormcv.deploy.CaptionerTopology' and global_perBatch:
        convert_batch = True
    else:
        convert_batch = False

update_stage_info('nl.tno.stormcv.deploy.DNNTopology', 5)


globpattern = None
def _globpattern():
    """Return the glob pattern"""
    if not globpattern is None:
        return globpattern
    return time.strftime("archive/%Y-%-m-%d/1")


def read_log(filename, pattern):
    """Read log entries from file"""
    with open(filename) as file:
        lines = file.readlines()
    return [pattern.match(line).groupdict() for line in lines if pattern.match(line)]


def correct_log_type(logs):
    """Correct log entry data types"""
    if skipOp:
        logs = [log for log in logs
                if log['evt'] != 'OpBegin' and log['evt'] != 'OpEnd']

    for log in logs:
        log['req'] = int(log['req'])
        log['seq'] = int(log['seq'])
        if log['stage'] == 'queue':
            log['stage'] = 'spout'
        elif log['stage'] == 'fetcher':
            log['stage'] = 'spout'
        if log['evt'] == 'OpBegin':
            log['stage'] = log['stage'][log['stage'].rfind('.')+1:]
        elif log['evt'] == 'OpEnd':
            log['stage'] = log['stage'][log['stage'].rfind('.')+1:]
        log['stamp'] = int(log['stamp'])
        if log['size'] is not None:
            log['size'] = int(log['size'])
        else:
            log['size'] = 0
        if log['batch'] is not None:
            log['batch'] = int(log['batch'])
        else:
            log['batch'] = -1
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
    def getter(frame_log_entry):
        """Inner"""
        keys = []
        for k in args:
            if k == 'stage':
                if frame_log_entry['stage'] in stages2idx:
                    keys.append(stages2idx[frame_log_entry[k]])
                else:
                    print('Unknown stage {}, full log: {}'.format(frame_log_entry['stage'],
                                                                  frame_log_entry))
                    keys.append(stages2idx['fat_features']+0.5)
            else:
                keys.append(frame_log_entry[k])
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
             res[idx + 1]['stamp'],
             res[idx + 2]['stamp']) = (res[idx + 2]['stamp'],
                                       res[idx + 2]['stamp'],
                                       res[idx]['stamp'])
            counter += 1
        elif (res[idx]['evt'] == 'Leaving' and
              res[idx - 1]['evt'] == 'Entering' and
              # important, we only handle cases that not handled in first clause
              res[idx - 2]['evt'] != 'Entering' and
              res[idx - 1]['stage'] == res[idx + 1]['stage'] and
              res[idx - 1]['seq'] == res[idx]['seq'] and
              res[idx]['seq'] == res[idx + 1]['seq']):
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


def load_gpu(filename):
    """Load gpu utilization data"""
    header_names = ['Date', 'Time', 'GPU ID', 'pwr', 'temp',
                    'sm', 'mem', 'enc', 'dec', 'mclk', 'pclk']
    df = pd.read_csv(filename, header=None, names=header_names, delim_whitespace=True, comment='#')
    return df.to_dict('records')


def collect_log(log_dir=None):
    """Collect log from files"""
    pttn = re.compile(r'^.+RequestID: (?P<req>[0-9-]+) StreamID: (?P<id>[^ ]+)'
                      r' SequenceNr: (?P<seq>\d+)'
                      r'( BatchId: (?P<batch>[0-9-]+))?'
                      r' (?P<evt>\w+) (?P<stage>[\w.]+):'
                      r' (?P<stamp>\d+)'
                      r'( Size: (?P<size>\d+))?$')
    logs = []
    cpus = {}
    gpus = {}
    if log_dir is None:
        log_dir = _globpattern()
    for machine in next(os.walk(log_dir))[1]:
        if storm102:
            print('the glob pattern is', os.path.join(log_dir, machine, '*', 'worker.log'))
            files = glob.glob(os.path.join(log_dir, machine, '*', 'worker.log'))
        else:
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

        if has_gpu_log:
            gpu_log = os.path.join(log_dir, machine, 'log.gpu')
            print('Collect cpu log from', gpu_log, file=sys.stderr)
            gpus[machine] = load_gpu(gpu_log)

    logs = correct_log_type(logs)
    streams_log = group_by(logs, 'id')

    streams = {}
    lagecy_tidy_logs = None
    counter = 0
    for per_stream in streams_log:
        tidy_logs, corrected_counter = zip(*[tidy_frame_logs(per_frame, debug=global_debug)
                                             for per_frame in group_by_frame(per_stream)])
        tidy_logs = list(tidy_logs)
        tidy_logs.sort(key=lambda per_frame_logs: per_frame_logs[0]['seq'])
        if lagecy_tidy_logs is None:
            lagecy_tidy_logs = tidy_logs
        stream_id = tidy_logs[0][0]['id']
        streams[stream_id] = tidy_logs
        counter += sum(corrected_counter)

    print('Auto fixed cross stage timming issues for {} log entries'.format(counter),
          file=sys.stderr)
    return streams, cpus, gpus, logs


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
                if evt == 'Entering' and trial[idx]['stage'].endswith('batcher'):
                    frame['enter-batch-stamp'] = trial[idx]['stamp']
                if trial[idx]['batch'] != -1:
                    if 'batch' in frame and frame['batch'] != trial[idx]['batch']:
                        print('WARNING: Frame {} has multiple batchId.'.format(frame['seq']),
                              file=sys.stderr)
                    if evt == 'Leaving' and trial[idx]['stage'].endswith('batcher'):
                        frame['batch'] = trial[idx]['batch']
                trial[idx] = (evt, trial[idx]['stage'], trial[idx]['stamp'])
            trial[:0] = head
        frame['retries'] = [item for item in retries if isinstance(item[0], tuple)]
        if 'batch' not in frame:
            frame['batch'] = -1
        if 'enter-batch-stamp' not in frame:
            frame['enter-batch-stamp'] = 0
            frame['batch'] = -1
        return frame

    frames = [extract_frame(frame_entries) for frame_entries in tidy_logs]
    return frames


def check_frame(frame, debug=False):
    """Sanity check a single frame"""
    for trial in frame['retries']:
        in_stage = False
        last_stage = None
        for evt, curr_stage, _ in trial:
            if evt == 'Entering' and not in_stage:
                if last_stage is not None and next_stage(last_stage) != curr_stage:
                    if debug:
                        print('Non consecutive stage for frame {}, last {}, current {}, should be {}'
                              .format(frame['seq'], last_stage, curr_stage, next_stage(last_stage)),
                              file=sys.stderr)
                    # Non consecutive stage
                    return False
                last_stage = curr_stage
                in_stage = True
            elif evt == 'Leaving' and in_stage:
                if last_stage != curr_stage:
                    # Non consecutive stage entering/leaving
                    if debug:
                        print('Non consecutive stage entering/leaving for frame', frame['seq'],
                              ': last', last_stage, 'leaving', curr_stage, file=sys.stderr)
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
            for evt, stage_name, stamp in trial[1:]:
                if evt == 'Entering':
                    if prev_stage(stage_name).endswith('batcher'):
                        # we know there's no latency between batcher and the following stage
                        continue
                    # Cross stage latency
                    latency = stamp - last_stamp
                    key = '{}-{}'.format(prev_stage(stage_name), stage_name)
                    latencies[key].append(latency)
                    last_stamp = stamp
                elif evt == 'Leaving':
                    # In stage latency
                    latency = stamp - last_stamp
                    latencies[stage_name].append(latency)
                    if stage_name == last_stage():
                        # The frame left the last stage, compute total latency
                        # start from left spout
                        if len(trial) <= 1:
                            # the frame never left spout, which should not happen
                            # when the code path execute to here, but just in case
                            latency = 0
                        else:
                            latency = stamp - trial[1][2]
                        latencies['total'].append(latency)
                    last_stamp = stamp
                elif evt == 'OpBegin':
                    last_op_stamp = stamp
                elif evt == 'OpEnd':
                    # In frameOp latency
                    latency = stamp - last_op_stamp
                    key = 'Op:{}'.format(stage_name)
                    latencies[key].append(latency)
                    last_op_stamp = stamp
                elif evt == 'Ack':
                    # Last bolt to spout ack
                    latency = stamp - last_stamp
                    key = '{}-{}'.format(prev_stage(stage_name), stage_name)
                    latencies[key].append(latency)
                    last_stamp = stamp
        if not frame['failed'] is None and 'total' in latencies:
            # The frame failed but left the final stage
            anomaly_cnt = anomaly_cnt + 1

        # Average on each latency item and add to global avg_latency
        for k in latencies.keys():
            avg_latency[k] = avg_latency[k] + latencies[k]
            latencies[k] = sum(latencies[k])/len(latencies[k])
        latencies = dict(latencies)
        latencies['service'] = 0
        try:
            for st in stages:
                if st != 'spout' and st != 'ack' and not st.endswith('batcher'):
                    latencies['service'] += latencies[st]
        except KeyError as e:
            # The frame didn't go through all stages
            del latencies['service']
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
        print('Stream: {} Machine: {:9} Seq: {:<5}\t{:<8} {:^20}: {}\tSize: {}'
              .format(item['id'], item['machine'], item['seq'], item['evt'], item['stage'],
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


def convert_to_batch(frames):
    """Convert a list of frames to a list of batches"""
    batches = []
    for framesInABatch in group_by(frames, 'batch'):
        framesInABatch.sort(key=frame_key_getter('enter-batch-stamp'))
        lastFrame = framesInABatch[-1]
        batch = copy(lastFrame)
        batch['seq'] = batch['batch']
        batches.append(batch)

    def update_batch(batchId, d):
        """Update a given batch"""
        for b in batches:
            if b['batch'] == batchId:
                b.update(d)

    def find_batch(batchId):
        """Update a given batch"""
        for idx in range(0, len(batches)):
            if batches[idx]['batch'] == batchId:
                return idx
        return -1


    # Complete captioner and streamer latency for batches
    for frame in frames:
        done = False
        for trial in frame['retries']:
            for evt, st, stamp in trial:
                if is_after(st, 'frame_grouper'):
                    # This frame is actually a GroupOfFrame object in java
                    idx = find_batch(frame['batch'])
                    # add extra latency values
                    for k, v in frame['latencies'].items():
                        if k not in batches[idx]['latencies']:
                            batches[idx]['latencies'][k] = v
                    # re-calc service and total latency
                    latencies = batches[idx]['latencies']
                    latencies['service'] = 0
                    try:
                        for st in stages:
                            if st != 'spout' and st != 'ack' and not st.endswith('batcher'):
                                latencies['service'] += latencies[st]
                    except KeyError as e:
                        # The frame doesn't go through all stages
                        latencies['service'] = -1
                    latencies['total'] = 0
                    for k, v in latencies.items():
                        if k != 'service' and k != 'total':
                            latencies['total'] += v

                    done = True
                    break;
            if done:
                break
    return batches


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


def cdf_plot(clean_frames, stage=None, label_format=None, **kwargs):
    """Plot CDF,
    from http://stackoverflow.com/questions/25577352/plotting-cdf-of-a-pandas-series-in-python
    """
    if stage is None:
        stage = 'total'
    if not isinstance(stage, list):
        stage = [stage]
    if label_format is None:
        label_format = '{stage}'

    p = None
    if 'ax' in kwargs:
        p = kwargs['ax']
        del kwargs['ax']
    for st in stage:
        ser = pd.Series([frame['latencies'][st] for frame in clean_frames if st in frame['latencies']])
        ser = ser.sort_values()
        ser[len(ser)] = ser.iloc[-1]
        cum_dist = np.linspace(0.,1.,len(ser))
        ser_cdf = pd.Series(cum_dist, index=ser)
        p = ser_cdf.plot(drawstyle='steps', ax=p, label=label_format.format(stage=st),
                         legend=True, **kwargs)
    p.set_xlabel('Latency (ms)')
    p.figure.tight_layout()
    return p


def seq_any_plot(clean_frames, getter, should_include, **kwargs):
    """Plot!"""
    ff = sorted(clean_frames, key=frame_key_getter('seq'))
    ff = [frame for frame in ff if should_include(frame)]

    df = pd.DataFrame({
        s: pd.Series([getter(frame, s) for frame in ff])
        for s in getter(None)
    })
    df.loc[:, 'SequenceNr'] = [frame['seq'] for frame in ff]
    thePlot = df.plot(x='SequenceNr', **kwargs)
    thePlot.set_xlabel('SequenceNr')
    thePlot.get_xaxis().set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    thePlot.figure.tight_layout()
    return thePlot


def time_latency_plot(clean_frames, stage='total', **kwargs):
    """Plot!"""
    if not isinstance(stage, list):
        stage = [stage]

    def getter(frame, s=None):
        """a getter"""
        if frame is None:
            return stage
        return frame['latencies'][s]

    def should_include(frame):
        """should include"""
        return set.issubset(set(stage), frame['latencies'].keys())

    thePlot = seq_any_plot(clean_frames, getter, should_include, **kwargs)
    thePlot.set_ylabel('Latency (ms)')
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
    streams, cpus, gpus, raw_logs = collect_log(log_dir)
    
    mpl.rcParams['figure.dpi'] = 193
    mpl.rcParams['axes.formatter.useoffset'] = False
    sns.plt.ion()
    sns.set_style('whitegrid')

    return streams, cpus, gpus, raw_logs

def post_process(streams, stream_id):
    """Post process log for a stream"""
    tidy_logs = streams[stream_id]

    frames = extract_frames(tidy_logs)
    clean_frames = sanity_filter(frames)
    compute_latency(clean_frames)
    latency_check(clean_frames)
    distributions = compute_stage_dist(tidy_logs)
    if convert_batch:
        # actually here the new clean_frames list represents a list of batches
        clean_frames = convert_to_batch(clean_frames)
    return tidy_logs, frames, clean_frames, distributions

class exp_res(object):
    """An object holds all parsed logs for one experiment"""
    def __init__(self, exp_name):
        self.exp_name = exp_name
        if exp_name.find('/') != -1:
            cut = dt(2016, 7, 18)
            cut2 = dt(2016, 9, 1)
            cut3 = dt(2016, 9, 14)
            cut4 = dt(2016, 9, 25)
            self.exp_date = dt.strptime(exp_name[:exp_name.find('/')], "%Y-%m-%d")
            if self.exp_date < cut:
                self.log_version = 1
            elif self.exp_date < cut2:
                self.log_version = 2
            elif self.exp_date < cut3:
                self.log_version = 3
            elif self.exp_date < cut4:
                self.log_version = 4
            else:
                self.log_version = 5
        else:
            self.log_version = 5

        log_dir = 'archive/{}'.format(exp_name)
        self._load_params(os.path.join(log_dir, 'params.txt'))
        self.params['exp_name'] = self.exp_name

        self._ensure_stage_info()

        lagecy_stream_id = None
        self.streams = {}
        streams_logs, self.cpus, self.gpus, self.raw_logs = run(log_dir)
        for stream_id, per_stream_tidy_log in streams_logs.items():
            if lagecy_stream_id is None:
                lagecy_stream_id = stream_id
            (tidy_logs, frames, clean_frames,
             distributions) = post_process(streams_logs, stream_id)
            self.streams[stream_id] = {
                'tidy_logs': tidy_logs,
                'frames': frames,
                'clean_frames': clean_frames,
                'distributions': distributions,
                'frame_filter': None
            }
        self.select_stream(lagecy_stream_id)

        # node count
        nodes = set()
        for log in flattened(self.tidy_logs):
            nodes.add(log['machine'])
        self.params['nodes'] = nodes

        self.seqs = sorted([f['seq'] for f in self.clean_frames])
        self.filters = None

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
        if 'fat_features' in self.stages:
            return '{fat}+{drawer}'.format(**self.params)
        elif self.params['topology_class'] == 'nl.tno.stormcv.deploy.BatchDNNTopology':
            return '{fat}+{drawer}'.format(**self.params)
        elif self.params['topology_class'] == 'nl.tno.stormcv.deploy.SpoutOnly':
            return '1+1'
        elif self.params['topology_class'] == 'nl.tno.stormcv.deploy.SplitDNNTopology':
            return '{facedetect}+{dnnforward}+{dnnclassify}+{drawer}'.format(**self.params)
        elif self.params['topology_class'] == 'nl.tno.stormcv.deploy.ObjTrackingTopology':
            return '1+{drawer}'.format(**self.params)
        elif self.params['topology_class'] == 'nl.tno.stormcv.deploy.LoopTopology':
            return '{scale}+1+{drawer}'.format(**self.params)
        elif self.params['topology_class'] == 'nl.tno.stormcv.deploy.E3_MultipleFeaturesTopology':
            return '{face}+{sift}+1+{drawer}'.format(**self.params)
        elif self.params['topology_class'] == 'nl.tno.stormcv.deploy.CaptionerTopology':
            return '{vgg}+1+{captioner}'.format(**self.params)
        else:
            return 'unknown'

    def _load_params(self, params_path):
        """Load parameters"""
        key_translate = {
            'cpu-per-node': 'cpu_per_node',
            'topo': 'topology_class',
            'topo-id': 'topo_id'
        }
        self.params = {}
        with open(params_path) as f:
            lines = [(line[:-1] if line.endswith('\n') else line) for line in f.readlines()]
            for line in lines:
                if line.startswith('--'):
                    key, value = line[2:].split('=')
                    self.params[key] = value
                elif '=' in line:
                    k, v = line.split('=')
                    if k in key_translate:
                        k = key_translate[k]
                    self.params[k] = v
                else:
                    # legacy
                    self.params['topology_class'] = line
        if 'topology_class' not in self.params:
            print('ERROR: topology_class not found!!!')
            raise TypeError('topology_class not found for {}'.format(self.exp_name))
        self.stages = topology_stage_map[self.params['topology_class']]
        # parse numbers
        for key in ['fps', 'scale', 'fat', 'drawer', 'batch-size']:
            if key in self.params:
                self.params[key] = int(self.params[key])

    def _ensure_stage_info(self):
        """Use correct stage info"""
        update_stage_info(self.params['topology_class'], self.log_version)

    def _sample(self, count=1):
        """Sample only middle part of the seqences"""
        trim_ratio = 0.25 if len(self.seqs) > 500 else 0.05
        st_idx = int(len(self.seqs) * trim_ratio)
        ed_idx = int(len(self.seqs) * (1 - trim_ratio))

        sample_space = self.seqs[st_idx:ed_idx]
        sample_len = int(len(sample_space) / count)

        sample_indecis = zip(range(st_idx, ed_idx, sample_len),
                             range(st_idx + sample_len, ed_idx + sample_len, sample_len))
        samples = []
        for s, e in sample_indecis:
            samples.append(self.seqs[s:e])
        return samples

    def _select(self, seq, raw=False, **kwargs):
        """Select a subset of frames using seq"""
        if seq is None:
            seq = self.seqs
        elif seq == 'sample':
            seq = flattened(self._sample(**kwargs))

        try:
            seq = set(seq)
        except TypeError:
            seq = set([seq])

        ff = self.frames if raw else self.clean_frames

        selected = [frame for frame in ff
                    if frame['seq'] in seq]
        return selected, seq

    def select_stream(self, stream_id):
        """Select current stream"""
        self.current_stream_id = stream_id
        self.params['current_stream_id'] = stream_id
        self.distributions = self.streams[stream_id]['distributions']

        self.tidy_logs = self.streams[stream_id]['tidy_logs']
        self.frames = self.streams[stream_id]['frames']
        self.clean_frames = self.streams[stream_id]['clean_frames']

        frame_filter = self.streams[stream_id]['frame_filter']
        if frame_filter is not None:
            (self.tidy_logs,
             self.frames,
             self.clean_frames) = frame_filter(self.tidy_logs,
                                               self.frames,
                                               self.clean_frames)

    def set_frame_filter(self, func, stream_id = None):
        if stream_id is None:
            stream_id = self.current_stream_id

        self.streams[stream_id]['frame_filter'] = func

        # re-select current stream, in case we are changing 
        # current stream frame filter
        self.select_stream(self.current_stream_id)

    def show_log(self, seq=None):
        """Print raw log entries"""
        self._ensure_stage_info()
        if seq is not None and not isinstance(seq, list):
            seq = [seq]
        elif seq is None:
            seq = self.seqs

        for s in seq:
            show_log(self.tidy_logs[s], s)

    def show_frame(self, seq=None, raw=True):
        """Print frame entries"""
        self._ensure_stage_info()
        for frame in self._select(seq, raw)[0]:
            print('Seq: {:<5}\tRetries: {:<2}\tFailed: {}'
                  .format(frame['seq'], len(frame['retries']), frame['failed']))
            for trial in frame['retries']:
                print('------------------')
                for evt, stage_name, stamp in trial:
                    print('{:<8} {:^12}: {}'.format(evt, stage_name, stamp))

    def param_core(self):
        """Get cores"""
        self._ensure_stage_info()
        return int(self.params['scale']) + int(self.params['fat']) + int(self.params['drawer']) + 3

    def latency(self, seq=None, title=None, **kwargs):
        """Print and plot selected frames. seq can be a list or a single number"""
        self._ensure_stage_info()
        frames, frame_seqs = self._select(seq)
        p = latency_plot(frames, **kwargs)
        if title is None:
            p.set_title('Average Latency for Frame: {}'.format(str_range(frame_seqs) if seq is not None else 'All'))
        else:
            p.set_title(title)
        p.figure.canvas.set_window_title('Exp: {}'.format(self.exp_name))
        p.figure.tight_layout()
        return p

    def cdf(self, seq=None, stage=None, **kwargs):
        """Plot CDF"""
        self._ensure_stage_info()
        frames, frame_seqs = self._select(seq)
        if 'label_format' in kwargs:
            kwargs['label_format'] = kwargs['label_format'].format(**self.params)
        p = cdf_plot(frames, stage, **kwargs)
        p.figure.canvas.set_window_title('Exp: {}'.format(self.exp_name))
        p.figure.tight_layout()
        return p

    def seq_plot(self, getter, should_include=None, seq=None, **kwargs):
        """Plot against seq"""
        self._ensure_stage_info()
        if should_include is None:
            should_include = lambda frame: True

        p = seq_any_plot(self._select(seq)[0], getter, should_include, **kwargs)
        p.figure.canvas.set_window_title('Exp: {}'.format(self.exp_name))
        p.figure.tight_layout()
        return p

    def seq_latency(self, stage='total', seq=None, **kwargs):
        """Plot latency to seq"""
        self._ensure_stage_info()
        p = time_latency_plot(self._select(seq)[0], stage, **kwargs)
        p.figure.canvas.set_window_title('Exp: {}'.format(self.exp_name))
        p.figure.tight_layout()
        return p

    def fps(self, point=None, step=1000, title=None, **kwargs):
        """Plot FPS at stage"""
        self._ensure_stage_info()
        if point is None:
            point = [('Entering', 'spout'), ('Leaving', 'streamer')]
        p = fps_plot(self.tidy_logs, point=point, step=step, **kwargs)
        if title is None:
            p.set_title('FPS')
        else:
            p.set_title(title)
        p.figure.canvas.set_window_title('Exp: {}'.format(self.exp_name))
        p.figure.tight_layout()
        return p

    def cpu(self, which=None, **kwargs):
        """Plot CPU usage"""
        self._ensure_stage_info()
        p = cpu_plot(self.cpus, which, **kwargs)
        p.canvas.set_window_title('Exp: {}'.format(self.exp_name))
        p.tight_layout()
        return p

    def avg_cpu(self, node=None, which=None, rg=None):
        self._ensure_stage_info()
        if which is None:
            which = 'average'
        if node is None:
            node = list(self.cpus.keys())[0]

        dfcpu = pd.DataFrame.from_records(self.cpus[node], index='timestamp')

        def predict(index):
            if rg is None:
                return True
            lb, ub = rg
            return (index >= lb) & (index < ub)

        return dfcpu.loc[predict(dfcpu.index)][which].mean()

    def avg_gpu(self, which, col, node=None):
        self._ensure_stage_info()
        if node is None:
            node = list(self.gpus.keys())[0]

        dfgpu = pd.DataFrame.from_records(self.gpus[node], index=['Date', 'Time'])
        dfgpu = dfgpu.loc[dfgpu['GPU ID'] == which]
        return dfgpu[col].mean()


    def avg_fps(self, stage='spout', evt='Entering', trim=False, limit=None, all_stream=False):
        """Calculate average fps"""
        self._ensure_stage_info()
        def calcfps():
            fpses = compute_fps(self.tidy_logs, stage, evt)
            if limit is not None:
                fpses = fpses[:limit]
            while trim and fpses[-1] == 0.0:
                fpses.pop()
            return fpses

        if all_stream:
            current = self.current_stream_id
            totalfps = []
            totallen = 0
            for sid in self.streams.keys():
                self.select_stream(sid)
                fpses = calcfps()
                totalfps.append(mean(fpses))
                totallen += len(fpses)
            self.select_stream(current)
            return (mean(totalfps), totallen)
        else:
            fpses = calcfps()
            return (mean(fpses), len(fpses))

    def avg_latency(self, which=None, useLinregress=True, detail=False):
        """Calculate average latency"""
        self._ensure_stage_info()
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
                       for f in self._select(self.seqs[begin_at:end_at])[0]
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
            samples.append(self._select(self.seqs[st_idx:mid_idx])[0])
            samples.append(self._select(self.seqs[mid_idx:ed_idx])[0])
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

    def filter(self, predict):
        """Return a subset of experiments"""
        res = cross_res()
        res.exps = [exp for exp in self.exps if predict(exp)]
        return res

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

    def fps(self, points=None, x=None, all_stream=False, **kwargs):
        """Average fps"""
        if points is None:
            points = [('Entering', 'spout'), ('Ack', 'ack')]
        df = pd.DataFrame({
            '{} {}'.format(evt, stage): pd.Series([exp.avg_fps(stage, evt,
                                                               trim=True, all_stream=all_stream)[0]
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

    def latency_cdf(self, stage=None, merge=True, **kwargs):
        """Latency CDF cross all exps. You should make sure this makes sence"""
        if merge:
            all_frames = []
            for exp in self.exps:
                all_frames += exp.clean_frames
            p = cdf_plot(all_frames, stage, **kwargs)
        else:
            p = None
            for exp in self.exps:
                p = exp.cdf(stage, ax=p, **kwargs)
            p.figure.canvas.set_window_title('CDF')
            p.figure.tight_layout()
        return p

    def plot_all(self):
        for exp in self.exps:
            fig, axs = sns.plt.subplots(ncols=2, figsize=(15,5))
            exp.latency(seq='sample', ax=axs[0])
            exp.fps(ax=axs[1])
            fig.suptitle(exp)
