# [SublimeLinter pylint-disable:not-context-manager ]
"""Common tasks to run experiments and collect logs"""
import contextlib
from fabric.api import task, settings, abort, run, cd, env, execute, runs_once, sudo, open_shell
from fabric.contrib.console import confirm

env.use_ssh_config = True

@contextlib.contextmanager
def tmux(session, destroy=False):
    """run commands in tmux session"""
    run('tmux new-session -Ad -s {}'.format(session))

    class tmux_session(object):
        """tmux session object"""
        def __init__(self, session):
            super(tmux_session, self).__init__()
            self.session = session
            self.cmd_pattern = 'tmux send-keys -t {} \'{{}}\' C-m'.format(session)
            self.neww_pattern = 'tmux new-window -at {}:{{}}'.format(session)
            self.attach_pattern = 'tmux attach-session -t {}'.format(session)
            self.sudoed = False

        def run(self, cmd):
            """Run commands in tmux"""
            run(self.cmd_pattern.format(cmd))

        def sudo(self, cmd):
            """Run commands using sudo"""
            if not self.sudoed:
                open_shell(self.attach_pattern)

        def new_window(self, name):
            """New window"""
            run(self.neww_pattern.format(name))

    try:
        yield tmux_session(session)
    finally:
        if destroy:
            run('tmux kill-session -C {}'.format(session))



@task
def host_type():
    """Run uname on remote hosts"""
    run('uname -s')
    #sudo('apt-get update')
    with tmux('test2') as session:
        session.run('sudo echo $SHELL')


base_dir = '/home/peifeng/VideoDB'
work_dir = '/home/peifeng/work'


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
        sudo('/usr/local/zookeeper-3.4.6/bin/zkServer.sh start')


@task
def run_exp():
    """Deploy"""
    execute(build)

