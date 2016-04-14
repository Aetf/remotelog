# [SublimeLinter pylint-disable:not-context-manager ]
"""Common tasks to run experiments and collect logs"""
import contextlib
import os.path
from fabric.api import (cd,
                        env,
                        execute,
                        hide,
                        open_shell,
                        run,
                        runs_once,
                        shell_env,
                        sudo,
                        task,
                       )

env.use_ssh_config = True

@contextlib.contextmanager
def tmux(session, cwd=None, destroy=False):
    """run commands in tmux session"""
    def remote_run(*args, **kwargs):
        """wrapper around fabric run"""
        return run(shell=False, *args, **kwargs)

    class tmux_session(object):
        """tmux session object"""
        def __init__(self, session, cwd):
            super(tmux_session, self).__init__()
            #tmux_cmd = 'tmux -CC'
            tmux_cmd = 'tmux'
            self.session = session
            self.cwd = cwd if cwd is not None else ''
            patterns = {
                'cmd': ['send-keys', '-t {se}{{wd}}{{pane}}',
                        "'cd {{cwd}}' C-m",
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
            cmd = cmd.replace("'", r"\'")
            return remote_run(self.patterns['cmd']
                              .format(wd=window, pane=pane, cwd=self.cwd, cmd=cmd))

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
        session.run('echo "I am at $PWD"', new_window='MyTest')
        with session.cd('/tmp'):
            session.run('echo "I am at $PWD"')
            with session.cd('/home/aetf/Downloads'):
                session.run('echo "I am at $PWD"')
                with session.cd('../bin'):
                    session.run('echo "I am at $PWD"')


base_dir = '/home/peifeng/VideoDB'
work_dir = '/home/peifeng/work'
runtime_dir = os.path.join(work_dir, 'run')


@task
def predeploy():
    """Pre deploy"""
    with cd(base_dir):
        run('git pull')


@task
@runs_once
def build():
    """Build"""
    execute(predeploy)
    with cd(base_dir):
        with cd('stormcv'):
            run('./gradlew install')
        with cd('stormcv-deploy'):
            run('mvn package')


@task
def start_zk():
    """Start zookeeper"""
    pass

@task
def start_storm():
    """Bring up storm servers"""
    with tmux('exp') as session:
        with shell_env(ZOOCFGDIR=os.path.join(runtime_dir, 'zookeeper', 'conf'),
                       ZOO_LOG_DIR=os.path.join(runtime_dir, 'zookeeper', 'data')):
            session.run('/usr/local/zookeeper-3.4.6/bin/zkServer.sh start')


@task
def run_exp():
    """Deploy"""
    execute(build)

