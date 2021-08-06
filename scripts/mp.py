from multiprocessing import Pool
import time

def square(i):
    time.sleep(0.01)
    return i ** 2

p = Pool(4)
nums = range(50)

start = time.time()
print ('Using imap')
for i in p.imap(square, nums):
    pass
print ('Time elapsed: %s' % (time.time() - start))

start = time.time()
print ('Using imap_unordered')
for i in p.imap_unordered(square, nums):
    pass
print ('Time elapsed: %s' % (time.time() - start))
