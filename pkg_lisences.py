import functools
import pkg_resources
import subprocess
import multiprocessing.pool

def get_pkg_license(pkg):
    try:
        lines = pkg.get_metadata_lines('METADATA')
    except:
        lines = pkg.get_metadata_lines('PKG-INFO')

    for line in lines:
        if line.startswith('License:'):
            return line[9:]
    return '(Licence not found)'

def get_pkg_home_page(pkg):
    cmd = "pip show {0} | awk '/^Home-page/{{print $2}}'".format(pkg.key)
    return subprocess.check_output(cmd, shell=True).decode('utf-8').strip()

def print_package_and_license(lock, pkg):
    delimiter = "|"
    line = f"{pkg.key}{delimiter}{pkg.version}{delimiter}{get_pkg_license(pkg)}{delimiter}{get_pkg_home_page(pkg)}"
    lock.acquire()
    print(line)
    lock.release()

def print_packages_and_licenses():
    lock = multiprocessing.Lock()
    print("Package|Version|License|Home Page")
    multiprocessing.pool. \
        ThreadPool(10). \
        map(functools.partial(print_package_and_license, lock),
            sorted(pkg_resources.working_set,
                   key=lambda x: x.key))

if __name__ == "__main__":
    print_packages_and_licenses()
