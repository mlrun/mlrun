import time

def noctx_job():
   print('im running')
   time.sleep(0.1)
   print('finish')

if __name__ == "__main__":
    noctx_job()
