import os
import threading
import _thread as thread

commands = []
# commands.append('python3 corr.py movie1 recon_160/')
# commands.append('python3 corr.py movie2 recon_160/')
# commands.append('python3 corr.py movie3 recon_160/')
# commands.append('python3 corr.py movie4 recon_160/')

commands.append('python3 corr.py rest1 recon_160/')
commands.append('python3 corr.py rest2 recon_160/')
commands.append('python3 corr.py rest3 recon_160/')
commands.append('python3 corr.py rest4 recon_160/')

commands.append('python3 corr.py retbar1 recon_160/')
commands.append('python3 corr.py retbar2 recon_160/')
commands.append('python3 corr.py retccw recon_160/')
commands.append('python3 corr.py retcw recon_160/')
commands.append('python3 corr.py retcon recon_160/')
commands.append('python3 corr.py retexp recon_160/')

for c in commands:
    os.system(c)

# average results and plot
def avg_commands(dir):
    c_str = 'avg_corr.py ' + dir
    os.system(c_str)

# p1 = Process(target=avg_commands, args=("/data_qnap/yifeis/ae_relu_trained_with_50/corr_latent/"))
# p2 = Process(target=avg_commands, args=("/data_qnap/yifeis/ae_relu_trained_with_50/corr_recon_80/"))
# p3 = Process(target=avg_commands, args=("/data_qnap/yifeis/ae_relu_trained_with_50/corr_recon_160/"))
# processes = [p1, p2, p3]
# for p in processes:
#     p.start()
# for p in processes:
#     p.join()
