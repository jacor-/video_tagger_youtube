import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
from os import listdir
from os.path import isfile, join
from support import *
import json
import wget
import multiprocessing
import requests
import time 
import os
from PIL import Image

'''
This script will download the images

You must fulfill the following variables:
path --> It is the path to the folder where images will be writen
summaries --> The summaries we will download, it can be (1, 2, 3, 4), by the moment we are using (1, 2, 3)
frames --> The frames by summary we want to download, by the moment we are using (0, 6, 12, 18)
'''

path = "/home/ubuntu/datasets_vilynx/uploaded_images/images_mult8"
summaries = "(1, 2, 3)"
frames = "(0, 6, 12, 18)"

def getCurrentFileNames():
  onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
  return onlyfiles

def downloadAndSaveImg(name, url):
  r = requests.get(url)
  open(path+"/"+name , 'wb').write(r.content)
  try:
    time.sleep(1)
    Image.open(path+"/"+name);
  except:
    os.remove(path+"/"+name)
    print ("IMAGE REMOVED "+ name)


def deleteDeadJobsAndWait(jobs, max_num_threads):
  while(len(jobs) >= max_num_threads):
    for job in jobs:
      if job.is_alive() == False:
        jobs.remove(job)
    time.sleep(0.2)
  return jobs

def waitJobs(jobs):
  for j2 in jobs:
    j2.join()
  return []

def main(max_num_threads = 20):
  current_files = getCurrentFileNames()

  [c, conn] = pg_connect()
  c.execute("SELECT hash, summary, position, url FROM frames_aux WHERE training_set_index = 1 and position in "+frames+" and summary in "+summaries+" order by hash, summary asc, position asc;")
  rows = c.fetchall()
  jobs = []
  for row, counter in zip(rows, range(len(rows))):
    #if 100.0*counter/len(rows) > 93:
    name = row[0]+"_"+str(row[1])+"_"+str(row[2])+".jpg"
    url = row[3]
    if counter%10 == 0: print str(100.0*counter/len(rows))+"% Completed"
    if name not in current_files:
      p = multiprocessing.Process(target=downloadAndSaveImg, args=(name,url))
      p.start()
      jobs.append(p)
      jobs = deleteDeadJobsAndWait(jobs, max_num_threads)

  waitJobs(jobs)
  conn.close()

main()
