import os
import time


def select_log_path(root='', name=''):
    """Select a location for the log file."""
    filename = '{0}.log'.format(name) if name else 'askcos.log'
    root = root or os.path.expanduser('~')
    log_path = os.path.join(root, filename)

    try:
        open(log_path, 'a').close()
    except OSError:
        log_path = os.path.join(os.getcwd(), filename)

    return log_path


class MyLogger:
    """
    Create logger. Four different levels of information output. A level 3 ("FATAL")
    log will exit the program.
    """
    log_file = select_log_path()
    levels = {
        0: 'INFO',
        1: 'WARN',
        2: 'ERROR',
        3: 'FATAL'
    }
    time_zero = 0

    @staticmethod
    def initialize_logFile(root='', name=''):
        """Clear previous log file and set initialization time."""
        if name:
            MyLogger.log_file = select_log_path(root, name)
        if os.path.isfile(MyLogger.log_file):
            os.remove(MyLogger.log_file)
        MyLogger.time_zero = time.time()

    @staticmethod
    def print_and_log(text, location, level=0):
        """Print message to stdout and write to log file."""
        time_elapsed = time.time() - MyLogger.time_zero

        tag = '{}@{}'.format(MyLogger.levels[level], location)[:25]
        outstr = '{:25s}: [{:04.3f}s]\t{}'.format(tag, time_elapsed, text)

        print(outstr)

        with open(MyLogger.log_file, 'a') as f:
            f.write(outstr)
            f.write('\n')

        if level == 3:
            quit()
