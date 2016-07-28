# [SublimeLinter pylint-disable:not-context-manager ]
"""Common tasks to run experiments and collect logs

DEPENDENCIES:
This script uses python package Fabric3, progressbar, which can be installed with command:
    pip install --user Fabric3 progressbar
It also requires tmux installed on remote server.

NOTE: adjust the configuration section and main_host/host_list below to suit your needs before
using it!

USAGE: fab run_exp:<configuration>,<topology>,<cpu-core>[,least][,additional-arguments]
USAGE: fab batch_run
run_exp just runs the experiment with specified arguments. What it does:
    * Check if you have uncommited changes in project dir on local machine
    * Kill previous topology (if any), stop storm, and delete previous log files
    * Update project from github, rebuild it if necessary
    * Disable additional cpu cores and only leave the ones requested
    * Start storm (zookeeper, nimbus, supervisor, ui)
    * Start cpu utilization monitor script
    * Submit topology to storm cluster
    * Wait for a while (configurable) before killing
    * Kill topology
    * Restore disabled cpu cores
    * Fetch logs from remote servers to local machine
batch_run just invoke run_exp multiple times with different argument combinations. You specify
arguments in a list of list, and batch_run will invoke run_exp with all possible combinations
of the arguments you specified. See batch_run definition for an example on how to use it.

<configuration>:        Set on which machine the topology is going to run.
                        'all' means run on clarity25 and clarity26 and use clarity26 as nimbus
                        'clarity25' means run only on clarity25
                        'clarity26' means run only on clarity26
                        See `main_host` and `host_list` for details.
<topology>:             The full qualified name of the topology class you are going to submit to
                        storm. If it's under package nl.tno.stormcv.deploy, you can only specify
                        its class name.
<cpu-core>:             The number of cpu cores you are going to use to run the topology. Other
                        cores are disabled during the experiment, and re-enabled when finishing.
                        Remember to change max_cpu_cores to suitable value.
[least]:                How long to wait before kill the topology in the experiment. Default value
                        is 5 (minutes).
[additional-arguments]: Addition arguments are passed directly to topology main function. These
                        arguments are passed as a plain string with '--' prepended as prefix. e.g.
                        'abc\\=123' would be '--abc=123' when received by the main function in
                        args. Note that you have to escape '=' when runing from command line like
                        in the example.
                        Refer to topology source code for available arguments.

EXAMPLE: fab run_exp:all,DNNTopology,32,'num-workers\\=2','fetcher\\=image','fps\\=14',\
         'max-spout-pending\\=10000','scale\\=2','fat\\=60','drawer\\=2','use-caffe\\=1',\
         'use-gpu\\=0'
The command says, run DNNTopology on all machines available (defined by host_list), with 2 workers
in total, using image fetcher, keeping FPS at 14, setting spout pending limit to 10000
(effectively disable spount automatic throttle), using 2 threads for scale bolt, 60 threads for
fat_features bolt, 2 threads for drawer bolt, using caffe for NN, not using GPU for computation.
"""
import contextlib
import pickle
import os.path
import time
import itertools
from fabric.api import (cd,
                        env,
                        execute,
                        hide,
                        hosts,
                        run,
                        runs_once,
                        settings,
                        shell_env,
                        sudo,
                        task,
                       )
from fabric import utils
from fabric.contrib.project import (rsync_project
                                   )
from progress.bar import IncrementalBar

current_milli_time = lambda: int(round(time.time() * 1000))
env.use_ssh_config = True

# ====================================================================
# Path Configurations
# ====================================================================

# path to VideoDB project on local machine
local_project = '/home/aetf/develop/vcs/VideoDB'
# path to VideoDB project on remote server
project_dir = '/home/peifeng/VideoDB'
# path to a work directory on remote server
work_dir = '/home/peifeng/work'

# path to storm installation on remote server
storm_path = '/home/peifeng/tools/storm'
# path to zookeeper installation on remote server
zookeeper_path = '/home/peifeng/tools/zookeeper'
# path to cpu accounting script on remote server
accounting_py = '/home/peifeng/work/accounting.py'

# path to input files on remote server
input_image = [
    '/home/peifeng/work/data/frame.320x240.jpg',
    #'/home/peifeng/work/data/frame.1080x1920.png',
]
input_video = [
    '/home/peifeng/work/data/Vid_A_ball.avi',
    #'/home/peifeng/work/data/Vid_I_person_crossing.avi',
]

# runtime path for zookeeper
runtime_dir = os.path.join(work_dir, 'run')
zoo_cfg_dir = os.path.join(runtime_dir, 'zookeeper', 'conf')
zoo_log_dir = os.path.join(runtime_dir, 'zookeeper', 'data')

# maximum cpu cores
max_cpu_cores = {
    'clarity24': 24,
    'clarity25': 32,
    'clarity26': 32,
}

#saved_params_file = 'saved_params.pickle'
saved_params_file = None

# ====================================================================
# Topology Info Configurations
# ====================================================================
topology_class2id = {
    'nl.tno.stormcv.deploy.SpoutOnly': 'spout_only',
    'nl.tno.stormcv.deploy.SplitDnnTopology': 'dnn_classify_split',
    'nl.tno.stormcv.deploy.ObjTrackingTopology': 'object_tracking',
    'nl.tno.stormcv.deploy.E4_SequentialFeaturesTopology': 'face_detection',
    'nl.tno.stormcv.deploy.E3_MultipleFeaturesTopology': 'feature_extraction',
    'nl.tno.stormcv.deploy.DNNTopology': 'dnn_classification',
    'nl.tno.stormcv.deploy.BatchDNNTopology': 'dnn_classification_batch',
    'nl.tno.stormcv.deploy.LoopTopology': 'simple_loop',
    'nl.tno.stormcv.deploy.CaptionerTopology': 'captioning'
}

def wait_with_progress(max_sec, msg=None, resolution=1):
    """Wait with a nice progress bar"""
    if msg is None:
        msg = 'Running'
    progbar = IncrementalBar(msg, max=max_sec * resolution,
                             suffix='%(percent).1f%% - %(elapsed_td)s')
    while progbar.elapsed < max_sec:
        time.sleep(1 / resolution)
        progbar.next()
    progbar.finish()

def main_host(configuration):
    """Get main host from configuration"""
    username = 'peifeng'
    if configuration is None:
        configuration = 'all'
    if configuration == 'all':
        return '{}@clarity26'.format(username)
    else:
        return '{}@{}'.format(username, configuration)

def host_list(configuration):
    """Get host list from configuration"""
    username = 'peifeng'
    l = ['{}@clarity25', '{}@clarity26']
    if configuration is None:
        configuration = 'all'
    if configuration == 'all':
        return [s.format(username) for s in l]
    else:
        return [main_host(configuration)]

@contextlib.contextmanager
def tmux(session, cwd=None, destroy=False, runner=None):
    """run commands in tmux session"""
    if runner is None:
        runner = run

    def remote_run(*args, **kwargs):
        """wrapper around fabric run"""
        return runner(shell=False, *args, **kwargs)

    class tmux_session(object):
        """tmux session object"""
        def __init__(self, session, cwd):
            super(tmux_session, self).__init__()
            #tmux_cmd = 'tmux -CC'
            tmux_cmd = 'tmux'
            self.session = session
            self.cwd = cwd if cwd is not None else ''
            self.environ = {}
            patterns = {
                'cmd': ['send-keys', '-t {se}{{wd}}{{pane}}',
                        "'cd {{cwd}}' C-m",
                        'env Space {{envvar}} Space',
                        "'{{cmd}}' C-m",
                       ],
                'neww': ['new-window', '-t {se} -n {{name}}'],
                'has': ['has-session', '-t {se}'],
                'new': ['new-session', '-d', '-s {se}'],
                'kill': ['kill-session', '-Ct {se}'],
                'killw': ['kill-window', '-t {se}:{{name}}'],
                'lsw': ['list-windows', '-F \'#W\'', '-t {se}'],
                'selectw': ['select-window', '-t {se}:{{name}}'],
            }
            self.patterns = {k: ' '.join([tmux_cmd] + v).format(se=session)
                             for k, v in patterns.items()}
            self._ensure_session()

        def _has_session(self):
            """has session"""
            return remote_run(self.patterns['has'], quiet=True)

        def _ensure_session(self):
            """Ensure session exists"""
            if self._has_session().failed:
                with hide('running', 'stdout', 'stderr'):
                    return remote_run(self.patterns['new'])
            return True

        def _kill_session(self):
            """Kill session"""
            if self._has_session().succeeded:
                with hide('running', 'stdout', 'stderr'):
                    return remote_run(self.patterns['kill'])
            return True

        def _kill_window(self, name):
            """Kill window"""
            with hide('running', 'stdout', 'stderr'):
                return remote_run(self.patterns['killw'].format(name=name), quiet=True)

        def _run_in_pane(self, cmd, window=None, pane=None):
            """Run commands in tmux window"""
            window = ':' + window if window is not None else ''
            pane = '.' + pane if window != '' and pane is not None else ''
            envvar = ' Space '.join(["{}= \"'\"'{}'\"'\"".format(k, v)
                                     for k, v in self.environ.items()])
            cwd = self.cwd

            cwd = cwd.replace("'", r"\'")
            cmd = cmd.replace("'", r"\'")
            return remote_run(self.patterns['cmd']
                              .format(wd=window, pane=pane, envvar=envvar, cwd=cwd, cmd=cmd))

        def _new_window(self, name):
            """Create a new window"""
            with hide('running', 'stdout', 'stderr'):
                print('Creating new window', name)
                return remote_run(self.patterns['neww'].format(name=name))

        def _list_windows(self):
            """List windows names in session"""
            with hide('running', 'stdout', 'stderr'):
                return remote_run(self.patterns['lsw']).split()

        def _select_window(self, name):
            """Select window as current window"""
            with hide('running', 'stdout', 'stderr'):
                return remote_run(self.patterns['selectw'].format(name=name))

        def run(self, cmd, new_window=None):
            """Run commands in tmux"""
            if new_window is not None:
                if new_window not in self._list_windows():
                    self._new_window(new_window)
                self._select_window(new_window)
            return self._run_in_pane(cmd, window=new_window)

        def kill(self, window=None):
            """Kill something"""
            if window is not None:
                return self._kill_window(window)

        def destroy(self):
            """Kill session"""
            return self._kill_session()

        @contextlib.contextmanager
        def cd(self, path):
            """Context manager for cd"""
            last_cwd = self.cwd
            if os.path.isabs(path):
                self.cwd = path
            else:
                self.cwd = os.path.normpath(os.path.join(self.cwd, path))
            try:
                yield
            finally:
                self.cwd = last_cwd

        @contextlib.contextmanager
        def env(self, clean_revert=False, **kwargs):
            """Set environment variables"""
            previous = {}
            new = []
            for key, value in kwargs.items():
                if key in self.environ:
                    previous[key] = self.environ[key]
                else:
                    new.append(key)
                self.environ[key] = value
            try:
                yield
            finally:
                if clean_revert:
                    for key, value in kwargs.items():
                        # If the current env value for this key still matches the
                        # value we set it to beforehand, we are OK to revert it to the
                        # pre-block value.
                        if key in self.environ and value == self.environ[key]:
                            if key in previous:
                                self.environ[key] = previous[key]
                            else:
                                del self.environ[key]
                else:
                    self.environ.update(previous)
                    for key in new:
                        del self.environ[key]

    ts = tmux_session(session, cwd)
    try:
        yield ts
    finally:
        if destroy:
            ts.destroy()


@task
def uptodate(proj=None):
    """If the project is up to date"""
    if proj is None:
        proj = project_dir
    with cd(proj):
        with hide('running', 'stdout'):
            run('git remote update')
            local = run('git rev-parse @')
            remote = run('git rev-parse @{u}')
            base = run('git merge-base @ @{u}')
        if local == remote:
            return True
        elif local == base:
            utils.warn('local: {} remote: {} base: {}'.format(local, remote, base))
            return False
        elif remote == base:
            utils.warn('local: {} remote: {} base: {}'.format(local, remote, base))
            utils.error('Push project first!!!')
        else:
            utils.warn('local: {} remote: {} base: {}'.format(local, remote, base))
            utils.error('local diverged!!!')
        return False

@task
@runs_once
def build(force=None):
    """Build"""
    force = force == 'True'
    if not force and uptodate():
        return
    with cd(project_dir):
        run('git pull')
        with cd('stormcv'):
            run('./gradlew install')
        with cd('stormcv-deploy'):
            run('mvn package')


@task
def zookeeper(action=None):
    """Bring up zookeeper servers"""
    if action is None:
        action = 'start'
    with shell_env(ZOOCFGDIR=zoo_cfg_dir,
                   ZOO_LOG_DIR=zoo_log_dir):
        run(zookeeper_path + '/bin/zkServer.sh ' + action)

@task
def storm_nimbus(action=None):
    """Bring up strom nimbus"""
    if action is None:
        action = 'start'
    with tmux('exp') as ts:
        if action == 'start':
            ts.run(storm_path + '/bin/storm nimbus', new_window='nimbus')
        elif action == 'stop':
            ts.kill(window='nimbus')
        else:
            print('unknown action: {}'.format(action))


@task
def storm_supervisor(action=None):
    """Bring up strom nimbus"""
    if action is None:
        action = 'start'
    with tmux('exp') as ts:
        if action == 'start':
            ts.run(storm_path + '/bin/storm supervisor', new_window='supervisor')
        elif action == 'stop':
            ts.kill(window='supervisor')
        else:
            print('unknown action: {}'.format(action))

@task
def storm_ui(action=None):
    """Bring up storm ui"""
    if action is None:
        action = 'start'
    with tmux('exp') as ts:
        if action == 'start':
            ts.run(storm_path + '/bin/storm ui', new_window='ui')
        elif action == 'stop':
            ts.kill(window='ui')
        else:
            print('unknown action: {}'.format(action))


@task
def storm_submit(topology, *args):
    """Submit jar to storm"""
    files = input_video
    for arg in args:
        if arg.startswith('fetcher'):
            fetcher = arg.split('=')[1]
            if fetcher == 'image':
                files = input_image
            elif fetcher == 'video':
                pass
            else:
                utils.error('unsupported fetcher')

    cmd = [
        storm_path + '/bin/storm',
        'jar',
        project_dir
            + '/stormcv-deploy/target/stormcv-deploy-0.0.1-SNAPSHOT-jar-with-dependencies.jar',
        topology,
    ]
    cmd += files
    cmd += ['--'+ arg for arg in args]
    run(' '.join(cmd))


@hosts('localhost')
@task
def storm(action=None, configuration=None):
    """Bring up storm cluster"""
    with hide('running'):
        execute(zookeeper, action=action, host=main_host(configuration))
        execute(storm_nimbus, action=action, host=main_host(configuration))
        execute(storm_supervisor, action=action, hosts=host_list(configuration))
        execute(storm_ui, action=action, hosts=main_host(configuration))


@task
def fetch_log(configuration, saved_params=None):
    """Fetch logs from server"""
    if saved_params is None:
        if os.path.exists(saved_params_file):
            with open(saved_params_file, 'rb') as f:
                saved_params = pickle.load(f)
        else:
            saved_params = []
            print('WARNING: Saved params not found')
    if len(saved_params) == 0:
        print('WARNING: Saved params is empty')
        params = {}
    else:
        params = saved_params.pop()

    log_dir = os.path.join(time.strftime("archive/%Y-%-m-%d/"), '{}-{}')
    num = 1
    while os.path.exists(log_dir.format(params['topo-id'], num)):
        num += 1
    log_dir = log_dir.format(params['topo-id'], num)
    os.makedirs(log_dir, exist_ok=True)

    if saved_params_file is not None:
        with open(saved_params_file, 'wb') as f:
            pickle.dump(saved_params, f)
    with open(os.path.join(log_dir, 'params.txt'), 'w') as f:
        for key in params.keys():
            if key == 'args':
                continue # handle args last
            if key == 'cpu':
                print('cpu-per-node={}'.format(params[key]), file=f)
            else:
                print('{}={}'.format(key, params[key]), file=f)
        for arg in params['args']:
            print('--' + arg, file=f)

    execute(pull_log_per_node, log_dir, hosts=host_list(configuration))


@task
def pull_log_per_node(log_dir):
    """pull log from node"""
    shorthost = env.host.replace('.eecs.umich.edu', '')
    per_machine_dir = os.path.join(log_dir, shorthost)

    rsync_project(remote_dir=storm_path + '/logs/*worker*.log*',
                  local_dir=per_machine_dir, upload=False)
    rsync_project(remote_dir=storm_path + '/logs/log.cpu',
                  local_dir=per_machine_dir, upload=False)


@task
def clean_log():
    """Clean logs from server"""
    cmd = [
        'rm',
        '-f',
        storm_path + '/logs/*worker*.log*',
        storm_path + '/logs/log.cpu'
    ]
    run(' '.join(cmd))


@task
def cpu_monitor(action=None):
    """Start/Stop cpu_monitor"""
    if action is None:
        action = 'start'
    with tmux('exp') as ts:
        if action == 'start':
            ts.run('python ' + accounting_py + ' ' + storm_path + '/logs', new_window='cpu')
        elif action == 'stop':
            ts.kill(window='cpu')
        else:
            print('unknown action: {}'.format(action))


@task
def limit_cpu(number=None):
    """Limit cpu cores to use"""
    shorthost = env.host.replace('.eecs.umich.edu', '')
    cpu_cores = max_cpu_cores[shorthost]

    if number is None:
        number = cpu_cores
    else:
        number = int(number)

    number = max(1, min(cpu_cores, number))
    cmdptrn = 'tee /sys/devices/system/cpu/cpu{}/online <<EOF\n{}\nEOF'
    with hide('running', 'stdout', 'stderr'):
        for i in range(1, number):
            sudo(cmdptrn.format(i, 1))
        for i in range(number, cpu_cores):
            sudo(cmdptrn.format(i, 0))


@task
def kill_topology(topology_id, wait_time=60):
    """Kill a running topology"""
    kill_cmd = [
        storm_path + '/bin/storm',
        'kill',
        str(topology_id),
        '-w',
        str(wait_time)
    ]
    query_cmd = [
        storm_path + '/bin/storm',
        'list',
    ]
    run(' '.join(kill_cmd))
    wait_with_progress(int(wait_time), 'Killing')
    killed = False
    while not killed:
        killed = True
        with hide('running', 'stdout', 'stderr'):
            for line in run(' '.join(query_cmd)).split():
                if topology_id in line:
                    killed = False
                    time.sleep(1)
                    break

@task
def kill_exp(configuration, topology_id=None, wait=1):
    """Kill running experiment"""
    if topology_id is not None:
        execute(kill_topology, topology_id=topology_id, wait_time=wait,
                host=main_host(configuration))
    execute(storm, action='stop', configuration=configuration)
    execute(cpu_monitor, action='stop', hosts=host_list(configuration))


@task
def run_exp(configuration=None, topology=None, cpu=None, *args, least=5):
    """Run experiment"""

    # save sudo password
    with settings(host_string=main_host(configuration)):
        with hide('running', 'stdout', 'stderr'):
            sudo('echo good')

    if not execute(uptodate, local_project, host='localhost')['localhost']:
        utils.error('Your working copy is not clean, which cannot be fetched by remote server')
        return

    if topology is None:
        topology = 'nl.tno.stormcv.deploy.SpoutOnly'
    else:
        if not topology.startswith('nl'):
            topology = 'nl.tno.stormcv.deploy.' + topology


    if cpu is None:
        cpu = max_cpu_cores
    else:
        cpu = int(cpu)

    least = int(least)

    topology_id = topology_class2id[topology]
    for arg in args:
        if arg.startswith('topology-id'):
            topology_id = arg.split('=')[1]

    if saved_params_file is not None and os.path.exists(saved_params_file):
        with open(saved_params_file, 'rb') as f:
            saved_params = pickle.load(f)
    else:
        saved_params = []
    saved_params.append({'topo': topology, 'topo-id': topology_id, 'args': args, 'cpu': cpu})

    with hide('stdout', 'stderr'):
        execute(kill_exp, configuration=configuration)
        execute(clean_log, hosts=host_list(configuration))

    execute(build, host=main_host(configuration))

    with hide('running', 'stdout'):
        execute(limit_cpu, cpu, hosts=host_list(configuration))

        execute(storm, action='start', configuration=configuration)
        execute(cpu_monitor, action='start', hosts=host_list(configuration))

    execute(storm_submit, topology, *args, host=main_host(configuration))

    wait_with_progress(60 * least, 'Running', resolution=2)

    with hide('stdout'):
        execute(kill_exp, topology_id=topology_id, wait=30,
                configuration=configuration)
        execute(limit_cpu, hosts=host_list(configuration))

    if saved_params_file is not None:
        with open(saved_params_file, 'wb') as f:
            pickle.dump(saved_params, f)

    print('saved_params is', saved_params)
    execute(fetch_log, saved_params=saved_params, configuration=configuration)


@task
def batch_run():
    """Batch run"""
    configuration = 'clarity26'
    topology = ['ObjTrackingTopology']
    cores = [32]
    args = [
        'num-workers=1',
        ['fetcher=image'],
        #['fps=25', 'fps=29', 'fps=30', 'fps=40', 'fps=50', 'fps=65', 'fps=80', 'fps=100'],
        #['fps=100', 'fps=100', 'fps=100', 'fps=100', 'fps=100'],
        ['fps=60', 'fps=63', 'fps=65'],
        #['fps=60'],
        'auto-sleep=0',
        'msg-timeout=1000000',
        'max-spout-pending=10000',
        'sliding-win=100',
        'sliding-wait=10',
        'force-single-frame=0',
        'roi=209,117,36,43',
        ['scale=4'],
        #['scale=1'],
        #['fat=27', 'fat=28', 'fat=29', 'fat=30', 'fat=31', 'fat=32'],
        #['fat=40', 'fat=50', 'fat=60', 'fat=80', 'fat=100'],
        #['fat=14', 'fat=16', 'fat=18', 'fat=20'],
        #['fat=80', 'fat=58', 'fat=68'],
        #['fat=80', 'fat=1005],
        #['objtrack=1',],
        'drawer=4'
        #'drawer=1'
    ]

    for idx, arg in enumerate(args):
        if not isinstance(arg, list):
            args[idx] = [arg]

    for combo in itertools.product(topology, cores, *args):
        print('combo: ', combo)
        execute(run_exp, configuration, least=2, *combo)


@task
def batch_run_gpu():
    """Batch run"""
    configuration = 'clarity24'
    topology = ['BatchDNNTopology']
    cores = [24]
    args = [
        'num-workers=1',
        'fetcher=image',
        'use-caffe=1',
        ['use-gpu=2'],
        #['batch-size=1', 'batch-size=2', 'batch-size=3', 'batch-size=4', 'batch-size=5'],
        ['batch-size=1', 'batch-size=2' ],
        #['batch-size=1'],
        #['fps=15', 'fps=20', 'fps=25', 'fps=30', 'fps=45'],
        #['fps=3', 'fps=4',],
        ['fps=50', 'fps=60', 'fps=70'],
        'auto-sleep=0',
        'msg-timeout=1000000',
        'max-spout-pending=10000',
        ['scale=3'],
        #['scale=1'],
        #['fat=27', 'fat=28', 'fat=29', 'fat=30', 'fat=31', 'fat=32'],
        #['fat=40', 'fat=50', 'fat=60', 'fat=80', 'fat=100'],
        #['fat=14', 'fat=16', 'fat=18', 'fat=20'],
        #['fat=80', 'fat=58', 'fat=68'],
        #['fat=80', 'fat=1005],
        ['fat=2',],
        'drawer=3'
        #'drawer=1'
    ]

    for idx, arg in enumerate(args):
        if not isinstance(arg, list):
            args[idx] = [arg]

    for combo in itertools.product(topology, cores, *args):
        print('combo: ', combo)
        execute(run_exp, configuration, least=2, *combo)


@task
def batch_run_cap():
    """Batch run"""
    configuration = 'clarity24'
    topology = ['CaptionerTopology']
    cores = [24]
    args = [
        'num-workers=1',
        'fetcher=image',
        ['use-gpu=2'],
        ['cap-use-gpu=2'],

        ['group-size=10', 'group-size=50', 'group-size=100'],
        #'min-group-size=10',
        #'max-group-size=200',

        #['fps=15', 'fps=20', 'fps=25'],
        #['fps=3', 'fps=4',],
        ['fps=15'],

        'auto-sleep=0',
        'msg-timeout=1000000',
        'max-spout-pending=10000',

        ['scale=3'],
        'vgg=10',
        'captioner=10'
    ]

    for idx, arg in enumerate(args):
        if not isinstance(arg, list):
            args[idx] = [arg]

    for combo in itertools.product(topology, cores, *args):
        print('combo: ', combo)
        execute(run_exp, configuration, least=2, *combo)


@task
def batch_run_spoutonly():
    """Batch run"""
    configuration = 'clarity26'
    topology = ['SpoutOnly']
    cores = [24]
    args = [
        'num-workers=1',
        'fetcher=image',

        ['fps=15', 'fps=50', 'fps=100'],

        'auto-sleep=0',
        'msg-timeout=1000000',
        'max-spout-pending=10000',
    ]

    for idx, arg in enumerate(args):
        if not isinstance(arg, list):
            args[idx] = [arg]

    for combo in itertools.product(topology, cores, *args):
        print('combo: ', combo)
        execute(run_exp, configuration, least=1, *combo)
