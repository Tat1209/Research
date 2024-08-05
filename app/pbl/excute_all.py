import subprocess


scripts = [
"/home/haselab/Documents/tat/Research/app/pbl/main_mnist.py",
# "/home/haselab/Documents/tat/Research/app/pbl/main_mnist copy.py",
"/home/haselab/Documents/tat/Research/app/pbl/main_cifar.py",
# "/home/haselab/Documents/tat/Research/app/pbl/main_cifar copy.py",
"/home/haselab/Documents/tat/Research/app/pbl/main_tiny.py",
# "/home/haselab/Documents/tat/Research/app/pbl/main_tiny copy.py",
]

for script in scripts:
    process = subprocess.Popen(['python', script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    stderr = process.communicate()[1]
    if stderr:
        print(f"Error executing {script}: {stderr}")