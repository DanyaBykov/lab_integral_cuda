from sys import argv
from subprocess import run
from statistics import stdev

def check_for_correctness(results):
    precision = 10**(-7)
    for res in results[1:]:
        if abs(res - results[0]) > precision:
            raise Exception("Results are not equal")

def runner(how_many_times):
    """
    1. Результат обчислення інтегралу.
    2. Досягнуту абсолютну похибку.
    3. Досягнуту відносну похибку.
    4. Мінімальний час виконання.
    5. Середній час виконання.
    6. Кореговане стандартне відхилення для вибірки (Corrected sample standard
    deviation) для часу виконання.
    """
    amount_of_meth = 3

    for i in range(1, amount_of_meth+1):
        results = []
        times = []
        abs_err = []
        rel_err = []
        for _ in range(int(how_many_times)):
            res = run(["./build/integrate_cuda", f"{i}", f"./cfg_files/func{i}.cfg"],capture_output=True, text=True)
            if res.returncode == 0:
                temp = res.stdout.split('\n')
                results.append(float(temp[0]))
                abs_err.append(float(temp[1]))
                rel_err.append(float(temp[2]))
                times.append(int(temp[3]))
            else:
                raise Exception(f"{res.stderr}. Program finished with code {res.returncode}")
        try:
            check_for_correctness(results)
        except Exception as e:
            print(e)
            continue
        print(results[0])
        print(min(abs_err))
        print(min(rel_err))
        print(min(times))
        print(sum(times)/len(times))
        print(stdev(times))
        print()

if __name__ == "__main__":
    from sys import argv
    if len(argv) < 2:
        print("Usage: python3 test_runner.py <how_many_times>")
    else:
        runner(argv[1])