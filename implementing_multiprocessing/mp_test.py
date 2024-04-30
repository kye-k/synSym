import time
import multiprocessing

def do_something(seconds):
    print(f'sleeping {seconds} second(s)...')
    time.sleep(seconds)
    print('Done sleeping...')

def main():
    start = time.perf_counter()

    processes = []

    for _ in range(10):
        p = multiprocessing.Process(target=do_something, args=[1.5])
        p.start()
        processes.append(p)
    
    for process in processes:
        process.join()

    finish = time.perf_counter()

    print(f'Finished in {round(finish-start, 2)} second(s)')

if __name__ == '__main__':
    main()