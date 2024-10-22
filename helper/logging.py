class Logging:
    def __init__(self, log_file):
        self.log_file = log_file

        # Clear the log file
        open(log_file, 'w').close()

        self.log = open(log_file, 'a')
    
    def log(self, args):
        self.log.write('----------------------\n')
        for arg in args:
            self.log.write(arg + '\n')
        self.log.write('----------------------\n')

    def close(self):
        self.log.close()