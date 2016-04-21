# [SublimeLinter pylint-disable:not-context-manager ]
"""Common tasks to run experiments and collect logs"""
import contextlib
import pickle
import os.path
import time
from fabric.api import (cd,
                        env,
                        execute,
                        hide,
                        hosts,
                        run,
                        runs_once,
                        settings,
                        shell_env,
                        task,
                       )
from fabric import utils
from fabric.contrib.project import (rsync_project
                                   )

env.use_ssh_config = True

saved_params_file = 'saved_params.pickle'
project_dir = '/home/peifeng/VideoDB'
work_dir = '/home/peifeng/work'
runtime_dir = os.path.join(work_dir, 'run')


def main_host(configuration):
    """Get main host from configuration"""
    username = 'peifeng'
    if configuration is None:
        return '{}@clarity25'.format(username)
    else:
        return '{}@{}'.format(username, configuration)

def host_list(configuration):
    """Get host list from configuration"""
    username = 'peifeng'
    l = ['{}@clarity25', '{}@clarity26']
    if configuration is None:
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
        run('git remote update')
        with hide('running', 'stdout', 'stderr'):
            local = run('git rev-parse @')
            remote = run('git rev-parse @{u}')
            base = run('git merge-base @ @{u}')
        if local == remote:
            return True
        elif local == base:
            return False
        elif remote == base:
            utils.error('Push project first!!!')
        else:
            utils.error('local diverged!!!')
        return False

@task
@runs_once
def build():
    """Build"""
    if uptodate():
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
    with shell_env(ZOOCFGDIR=os.path.join(runtime_dir, 'zookeeper', 'conf'),
                   ZOO_LOG_DIR=os.path.join(runtime_dir, 'zookeeper', 'data')):
        run('/usr/local/zookeeper-3.4.6/bin/zkServer.sh ' + action)

@task
def storm_nimbus(action=None):
    """Bring up strom nimbus"""
    if action is None:
        action = 'start'
    with tmux('exp') as ts:
        if action == 'start':
            ts.run('/home/peifeng/storm-0.10.0/bin/storm nimbus', new_window='nimbus')
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
            ts.run('/home/peifeng/storm-0.10.0/bin/storm supervisor', new_window='supervisor')
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
            ts.run('/home/peifeng/storm-0.10.0/bin/storm ui', new_window='ui')
        elif action == 'stop':
            ts.kill(window='ui')
        else:
            print('unknown action: {}'.format(action))


@task
def storm_submit(topology, *args):
    """Submit jar to storm"""
    cmd = [
        '/home/peifeng/storm-0.10.0/bin/storm',
        'jar',
        '/home/peifeng/work/stormcv-deploy-0.0.1-SNAPSHOT-jar-with-dependencies.jar',
        topology,
        '/home/peifeng/work/Breaking_Dawn_Part2_trailer.mp4',
    ]
    cmd += ['--'+ arg for arg in args]
    run(' '.join(cmd))


@hosts('localhost')
@task
def storm(action=None, configuration=None):
    """Bring up storm cluster"""
    execute(zookeeper, action=action, host=main_host(configuration))
    execute(storm_nimbus, action=action, host=main_host(configuration))
    execute(storm_supervisor, action=action, hosts=host_list(configuration))
    execute(storm_ui, action=action, hosts=main_host(configuration))


@task
def fetch_log():
    """Fetch logs from server"""
    log_dir = os.path.join(time.strftime("archive/%Y-%-m-%d/"), '{}')
    shorthost = env.host.replace('.eecs.umich.edu', '')
    per_machine_dir = os.path.join(log_dir, shorthost)


    num = 1
    while os.path.exists(per_machine_dir.format(num)):
        num += 1
    per_machine_dir = per_machine_dir.format(num)
    log_dir = log_dir.format(num)

    os.makedirs(log_dir, exist_ok=True)

    rsync_project(remote_dir='/home/peifeng/storm-0.10.0/logs/*worker*.log*',
                  local_dir=per_machine_dir, upload=False)
    rsync_project(remote_dir='/home/peifeng/storm-0.10.0/logs/log.cpu',
                  local_dir=per_machine_dir, upload=False)

    if os.path.exists(saved_params_file):
        with open(saved_params_file, 'rb') as f:
            saved_params = pickle.load(f)
    else:
        saved_params = []
        print('WARNING: Saved params not found')
    if len(saved_params) == 0:
        print('WARNING: Saved params is empty')
        params = []
    else:
        params = saved_params.pop()
    with open(saved_params_file, 'wb') as f:
        pickle.dump(saved_params, f)
    with open(os.path.join(log_dir, 'params.txt'), 'w') as f:
        print(params['topo'], file=f)
        for arg in params['args']:
            print('--' + arg, file=f)


@task
def clean_log():
    """Clean logs from server"""
    cmd = [
        'rm',
        '-f',
        '/home/peifeng/storm-0.10.0/logs/*worker*.log*',
        '/home/peifeng/storm-0.10.0/logs/log.cpu'
    ]
    run(' '.join(cmd))


@task
def cpu_monitor(action=None):
    """Start/Stop cpu_monitor"""
    if action is None:
        action = 'start'
    with tmux('exp') as ts:
        if action == 'start':
            ts.run('python /home/peifeng/work/accounting.py', new_window='cpu')
        elif action == 'stop':
            ts.kill(window='cpu')
        else:
            print('unknown action: {}'.format(action))

@task
def kill_exp(configuration):
    """Kill running experiment"""
    execute(storm, action='stop', configuration=configuration)
    execute(cpu_monitor, action='stop', hosts=host_list(configuration))


@task
def run_exp(configuration=None, topology=None, *args):
    """Run experiment"""

    if not execute(uptodate, '/home/aetf/develop/vcs/VideoDB', host='localhost')['localhost']:
        utils.error('Your working copy is not clean, which cannot be fetched by remote serves')
        return

    if topology is None:
        topology = 'nl.tno.stormcv.deploy.SpoutOnly'
    else:
        if not topology.startswith('nl'):
            topology = 'nl.tno.stormcv.deploy.' + topology

    if os.path.exists(saved_params_file):
        with open(saved_params_file, 'rb') as f:
            saved_params = pickle.load(f)
    else:
        saved_params = []
    print('Saving args: ', args)
    saved_params.append({'topo': topology, 'args': args})

    execute(kill_exp, configuration=configuration)
    execute(clean_log, host=main_host(configuration))

    execute(build, host=main_host(configuration))


    execute(storm, action='start', configuration=configuration)
    execute(cpu_monitor, action='start', hosts=host_list(configuration))

    execute(storm_submit, topology, *args, host=main_host(configuration))

    with open(saved_params_file, 'wb') as f:
        pickle.dump(saved_params, f)
