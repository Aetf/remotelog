# [SublimeLinter pylint-disable:not-context-manager ]
"""Common tasks to run experiments and collect logs"""
import contextlib
import os.path
from fabric.api import (cd,
                        env,
                        execute,
                        hide,
                        hosts,
                        open_shell,
                        run,
                        runs_once,
                        task,
                       )

env.use_ssh_config = True

@contextlib.contextmanager
def tmux(session, cwd=None, destroy=False, runner=None):
    """run commands in tmux session"""
    if runner is None:
        runner = run

    def remote_run(*args, **kwargs):
        """wrapper around fabric run"""
        return runner(shell=False, shell_escape=False, *args, **kwargs)

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
                'attach': ['attach-session', ' -t {se}'],
                'has': ['has-session', '-t {se}'],
                'new': ['new-session', '-d', '-s {se}'],
                'kill': ['kill-session', '-Ct {se}'],
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
            """Run commands in new window"""
            with hide('running', 'stdout', 'stderr'):
                return remote_run(self.patterns['neww'].format(name=name))

        def _list_windows(self):
            """List windows names in session"""
            with hide('running', 'stdout', 'stderr'):
                return remote_run(self.patterns['lsw']).split()

        def _select_window(self, name):
            """Select window as current window"""
            with hide('running', 'stdout', 'stderr'):
                return remote_run(self.patterns['selectw'].format(name=name))

        def _attach(self):
            """Attach"""
            return remote_run(self.patterns['attach'])

        def run(self, cmd, new_window=None):
            """Run commands in tmux"""
            if new_window is not None:
                if new_window not in self._list_windows():
                    self._new_window(new_window)
                self._select_window(new_window)
            return self._run_in_pane(cmd, window=new_window)

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
def host_type():
    """Run uname on remote hosts"""
    run('uname -s')
    #sudo('apt-get update')
    with tmux('test2') as session:
        #session.run('echo I am here')
        session.run('echo "I am at $PWD"', new_window='EnvTest')
        with session.cd('/tmp'):
            session.run('echo "I am at $PWD"')
            with session.cd('/home/aetf/Downloads'):
                session.run('echo "I am at $PWD"')
                with session.cd('../bin'):
                    session.run('echo "I am at $PWD"')
                with session.env(ATEST='heiheihei'):
                    session.run('echo "I am back at $PWD" now, $ATEST')
            session.run('echo "I am back at $PWD" again, $ATEST')


project_dir = '/home/peifeng/VideoDB'
work_dir = '/home/peifeng/work'
runtime_dir = os.path.join(work_dir, 'run')


@task
def predeploy():
    """Pre deploy"""
    with cd(project_dir):
        run('git pull')


@task
@runs_once
def build():
    """Build"""
    execute(predeploy)
    with cd(project_dir):
        with cd('stormcv'):
            run('./gradlew install')
        with cd('stormcv-deploy'):
            run('mvn package')


@hosts('peifeng@clarity25')
@task
def zookeeper(action=None):
    """Bring up storm servers"""
    if action is None:
        action = 'start'
    with tmux('exp') as ts:
        with ts.env(ZOOCFGDIR=os.path.join(runtime_dir, 'zookeeper', 'conf'),
                    ZOO_LOG_DIR=os.path.join(runtime_dir, 'zookeeper', 'data')):
            ts.run('/usr/local/zookeeper-3.4.6/bin/zkServer.sh ' + action)


@task
def run_exp():
    """Deploy"""
    execute(build)


@task
def attach(session_name=None):
    """Attach to exp session"""
    if session_name is None:
        session_name = 'exp'
    with tmux(session_name, runner=open_shell) as ts:
        ts.attach()
