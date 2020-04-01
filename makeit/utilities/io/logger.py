import os
import time
import makeit.global_config as gc


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
    '''
    Create logger. Four different levels of information output. A level 3 ("FATAL")
    log will exit the program.
    '''
    logFile = select_log_path()
    levels = {
        0: 'INFO',
        1: 'WARN',
        2: 'ERROR',
        3: 'FATAL'
    }

    @staticmethod
    def initialize_logFile(root='', name=''):
        if name:
            MyLogger.logFile = select_log_path(root, name)
        if os.path.isfile(MyLogger.logFile):
            os.remove(MyLogger.logFile)
        gc.time_zero = time.time()

    @staticmethod
    def print_and_log(text, location, level=0):
        file = open(MyLogger.logFile, 'a')
        time_elapsed = time.time() - gc.time_zero

        outstr = '{:25s}: [{:04.3f}s]\t{}'.format('{}@{}'.format(MyLogger.levels[level], location)[:25],
            time_elapsed, text)
        print(outstr)
        file.write(outstr)
        file.write('\n')
        file.close()
        if level == 3:
            quit()
