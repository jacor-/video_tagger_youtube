
#!/usr/bin/python

import urllib
import requests

videos_per_criteria = 72
api_key = "AIzaSyDV_m-yM9Fdba2rem6w6Cy2GJeWwE2_3r8"
search_criteria = ["dog", "cat", "bird", "turtle", 
                   "fight", "run", "walk", "climb", 
                   "messi", "neymar", "ronaldo", "guardiola", "mourinho",
                   "barcelona", "london", "rome", "amsterdam", "europe", "world",
                   "plane", "car", "bike",
                   "mariano rajoy", "pablo iglesias", "pedro sanchez", "albert rivera"
                   ]

def get_video_details(videos_info, videos, tags):
  for item in videos_info:
      videoid = item['id']['videoId']
      req = requests.get("https://www.googleapis.com/youtube/v3/videos?key={key}&fields=items(snippet(tags))&part=snippet&id={video_id}".format(video_id = videoid, key = api_key))
      try:
        tag = eval(req.content)['items'][0]['snippet']['tags']
      except:
        tag = []
      tags.append(list(tag))
      videos.append(videoid)
  return videos, tags


def collect_videos(search_criteria, videos_per_criteria):
  videos, tags = [], []
  
  p_search_criteria = urllib.quote(search_criteria)
  page_token = ""
  if videos_per_criteria > 50:
    videos_per_call = 50
  else:
    videos_per_call = videos_per_criteria

  req = requests.get("https://www.googleapis.com/youtube/v3/search?key={key}&fields=nextPageToken,items(id(videoId))&part=id,snippet&q={query}&maxResults={num_vids}{pageToken}".format(query = p_search_criteria, key = api_key, pageToken = page_token, num_vids = str(videos_per_call)))
  retrieved_videos = eval(req.content)

  # Collect tags for the videos in this page  
  videos, tags = get_video_details(retrieved_videos['items'], videos, tags)

  # If we do not have enough videos, go to the next page
  while len(videos) < videos_per_criteria:
    page_token = retrieved_videos['nextPageToken']
    req = requests.get("https://www.googleapis.com/youtube/v3/search?key={key}&fields=nextPageToken,items(id(videoId))&part=id,snippet&q={query}&maxResults={num_vids}{pageToken}".format(query = p_search_criteria, key = api_key, pageToken = "&pageToken="+page_token, num_vids = str(videos_per_call)))
    retrieved_videos = eval(req.content)
    videos, tags = get_video_details(retrieved_videos['items'], videos, tags)

  return videos[:videos_per_criteria], tags[:videos_per_criteria]

print("Expected " + str(len(search_criteria) * videos_per_criteria) + " videos")

vid_id, tags = [],[]
for to_search in search_criteria:
  print(len(vid_id))
  videos, tag = collect_videos(to_search, videos_per_criteria)
  vid_id += list(videos)
  tags += list(tag)
  break



#youtube-dl -o out https://www.youtube.com/watch?v=GF60Iuh643I