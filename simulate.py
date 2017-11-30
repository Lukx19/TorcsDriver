from __future__ import print_function


import os
import pickle
import subprocess
import os.path
import sys
import datetime
from shutil import copyfile
import traceback
import time
import signal
# import glob

# sys.path.insert(0, '../')


#sys.path.insert(0, os.path.dirname(fitness_implementation))

#exec('from ' + os.path.filename(fitness_implementation) + ' import fitness')

class TorcsFitnessEvaluation:
    FILE_PATH = os.path.realpath(__file__)
    DIR_PATH = os.path.dirname(FILE_PATH)

    CLIENT_PATH = os.path.join(DIR_PATH, '')
    SIGTERMHUTDOWN_WAIT = 10
    RESULT_SAWING_WAIT = 1
    SHUTDOWN_WAIT = 3

    def __init__(self, torcs_config, clients, debug_path="debug", timelimit=3):
        self.torcs_config = torcs_config
        self.clients = clients
        self.debug_path = debug_path
        if not os.path.exists(debug_path):
            os.makedirs(debug_path)
        self.server_timelimit = timelimit

    def _killServer(self, server):
        if server is not None:
            print('Killing server and its children')
        os.killpg(os.getpgid(server.pid), signal.SIGTERM)

    def _killClient(self, client):
        print('Terminating client' + str(client.pid))
        # Try to be gentle
        os.killpg(os.getpgid(client.pid), signal.SIGTERM)
        # give it some time to stop gracefully
        client.wait(timeout=self.SHUTDOWN_WAIT)
        # if it is still running, kill it
        if client.poll() is None:
            print('\tTrying to hard kill client')
            os.killpg(os.getpgid(client.pid), signal.SIGKILL)
            time.sleep(self.SHUTDOWN_WAIT)

    def _startClient(self, port, files, model_file):
        #'-o', results_file
        client = subprocess.Popen(
            ['python', 'run.py', '-p',
                str(port), '-o', files['result_file'], '-m', 'NEAT', '-f', model_file],
            stdout=files['stdout'], stderr=files['stderr'], cwd=self.CLIENT_PATH, preexec_fn=os.setsid)
        print('Started Client ' + str(client.pid))
        return client

    def _startServer(self, configuration, stdout, stderr):
        try:
            print('Starting server')
            # '-r', os.path.join(self.DIR_PATH, configuration)
            server = subprocess.Popen(
                ['time', 'torcs', '-nofuel', '-nolaptime', '-r',
                    os.path.join(self.DIR_PATH, configuration)],
                stdout=stdout,
                stderr=stderr,
                preexec_fn=os.setsid
            )
            print('Waiting for server to stop')
            server.wait(timeout=self.server_timelimit)

        except subprocess.TimeoutExpired:
            print('SERVER TIMED-OUT!')
            return False, server
        except:
            print('Ops! Something happened"')
            traceback.print_exc()
            return False, server
        return True, server

    def _openDebugFiles(self, base_path, folder):
        base_dir = str(os.path.join(base_path, folder))
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        files = {
            'stdout_file': os.path.join(base_dir, 'out.log'),
            'stderr_file': os.path.join(base_dir, 'err.log'),
            'result_file': os.path.join(base_dir, 'result.log'),
        }
        files['stdout'] = open(files['stdout_file'], 'w')
        files['stderr'] = open(files['stderr_file'], 'w')
        return files

    def _closeDebugFiles(self, files, time):
        files['stdout'].close()
        files['stderr'].close()

        copyfile(files['stdout_file'], files['stdout_file'] + str(time))
        copyfile(files['stderr_file'], files['stderr_file'] + str(time))
        copyfile(files['result_file'], files['result_file'] + str(time))

    def _parseResultFile(self, file_name):
        try:
            results = open(file_name, 'r')
            values = []
            skip_header = True
            for line in results.readlines():
                if skip_header == True:
                    skip_header = False
                    continue
                # read the comma-separated values in the first line of the file
                splited = [float(x) for x in line.split(',')]
                if len(splited) > 1:
                    values.append(splited)
            results.close()
        except IOError:
            # if the files doesn't exist print, there might have been some error...
            # print the stacktrace and return None
            print("Can't find the result file!")
            traceback.print_exc()
            values = None
        return values

    def evaluate(self, clients_model):
        current_time = datetime.datetime.now().isoformat()
        start_time = time.time()
        clients_data = []
        id = 1
        if len(clients_model) != len(self.clients):
            print("Not same number of input client data and client models in evaluate")
            return

        for i, client in enumerate(self.clients):
            c = {}
            model_file = self.debug_path + '/' + \
                'model_temp' + str(id) + '.pickle'

            # dumping model to file so it can be used inside driver
            with open(model_file, 'wb') as f:
                pickle.dump(clients_model[i], f)

            c['files'] = self._openDebugFiles(
                self.debug_path, 'client' + str(id))
            c['client'] = self._startClient(
                port=client['port'], model_file=model_file, files=c['files'])
            print('Results at', c['files']['result_file'])
            id += 1
            clients_data.append(c)
        server_debugs = self._openDebugFiles(self.debug_path, 'server')
        state, server = self._startServer(configuration=self.torcs_config,
                                          stdout=server_debugs['stdout'], stderr=server_debugs['stderr'])
        if state == False:
            self._killServer(server)
        for client in clients_data:
                # copyfile(client['model_file'], client['model_file'] + 'error')
            self._killClient(client['client'])
            self._closeDebugFiles(client['files'], current_time)

        print('Simulation ended')

        # wait a second for the results file to be created

        # time.sleep(30)

        # if the result file hasn't been created yet, try 10 times waiting 'RESULT_SAWING_WAIT' seconds between each attempt

        results = []
        for client in clients_data:
            results.append(self._parseResultFile(
                client['files']['result_file']))
        end_time = time.time()

        print('Total Execution Time =', end_time - start_time, 'seconds')

        return results


# def clean_temp_files(results_path, models_path):

#     print('Cleaning directories')

#     for zippath in glob.iglob(os.path.join(DIR_PATH, results_path, 'results_*')):

#         os.remove(zippath)

#     for zippath in glob.iglob(os.path.join(DIR_PATH, models_path, '*')):

#         os.remove(zippath)


# def initialize_experiments(
#     output_dir,
#     configuration=None,
#     port=3001):

#     results_path = os.path.join(output_dir, 'results')
#     models_path = os.path.join(output_dir, 'models')
#     debug_path = os.path.join(output_dir, 'debug')
#     checkpoints_path = os.path.join(output_dir, 'checkpoints')
#     directories = [checkpoints_path,
#                    os.path.join(debug_path, 'client'),
#                    os.path.join(debug_path, 'server'),
#                    models_path,
#                    results_path]

#     for d in directories:
#         if not os.path.exists(d):
#             os.makedirs(d)

#     if configuration is None:
#         configuration = os.path.join(output_dir, 'configuration.xml')

#     configuration = os.path.realpath(configuration)
#     debug_path = os.path.realpath(debug_path)
#     results_path = os.path.realpath(results_path)
#     models_path = os.path.realpath(models_path)
#     if not os.path.isfile(configuration):
#         print('Error! Configuration file "{}" does not exist in {}'.format(configuration))

#         raise FileNotFoundError(
#             'Error! Configuration file "{}" does not exist in {}'.format(configuration))

#     eval = lambda model: evaluate(model, configuration=configuration, port=port,debug_path=debug_path,results_path=results_path, models_path=models_path)

#     return results_path, models_path, debug_path, checkpoints_path, eval
