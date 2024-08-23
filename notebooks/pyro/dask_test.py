import numpy as np
import time
import torch
from dask.distributed import Client, wait, LocalCluster,  print

from tqdm import tqdm



def segment_fake(image):
    start = time.time()
    img = torch.from_numpy(image).to('cuda:0')
    img = img + 100.0
    time.sleep(0.5)
    duration = time.time() - start
    print(f"Segment fake takes: {duration}s")
    return img.cpu().numpy()

def barcodes_fake(image):
    start = time.time()
    img = torch.from_numpy(image).to('cuda:0')
    img = img + 10.0
    time.sleep(0.5)
    duration = time.time() - start
    print(f"Barcodes fake takes: {duration}s")
    return img.cpu().numpy()


def dots_fake(image):
    start = time.time()
    result = image - 100.0
    time.sleep(0.5)
    duration = time.time() - start
    print(f"Dots fake takes: {duration}s")
    return result

def internal_fake(segmented_image, dots_data, barcodes_data):
    start = time.time()
    time.sleep(0.5)
    _ = segmented_image + dots_data + barcodes_data
    duration = time.time() - start
    print(f"Internal fake takes: {duration}s")
    return True

if __name__ == '__main__':
    cluster = LocalCluster(n_workers=3, threads_per_worker=2)
    client = Client(cluster)
    
    results = []
    start = time.time()
    tasks = 100
    for i in tqdm(range(tasks)):
        img1 = np.random.random((800, 2400))
        image1 = client.scatter(img1)
        img2 = np.random.random((800, 2400))
        image2 = client.scatter(img2)
        seg_result = client.submit(segment_fake, image1)
        dots_result = client.submit(dots_fake, image2)
        barcodes_result = client.submit(barcodes_fake, image1)
        internal_result = client.submit(internal_fake, seg_result, dots_result, barcodes_result)
        #fire_and_forget(internal_result)
        results.append(internal_result)
        time.sleep(0.5)
        del image1, image2

    #for future, result in as_completed(results, with_results=True):
    #    print(result)
    wait(results)
    del seg_result, dots_result, barcodes_result, internal_result, results
    #wait(results)
    duration = time.time() - start
    print(f"Duration: {duration}s for each: {duration/tasks}")
    time.sleep(5)