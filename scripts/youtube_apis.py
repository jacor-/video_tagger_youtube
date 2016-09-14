
#!/usr/bin/python

import urllib
import requests

api_key = "AIzaSyDV_m-yM9Fdba2rem6w6Cy2GJeWwE2_3r8"
search_criteria = ["dog", "cat", "bird", "turtle", 
                   "fight", "run", "walk", "climb", 
                   "messi", "neymar", "ronaldo", "guardiola", "mourinho",
                   "barcelona", "london", "rome", "amsterdam", "europe", "world",
                   "plane", "car", "bike",
                   "mariano rajoy", "pablo iglesias", "pedro sanchez", "albert rivera"
                   ]

def collect_videos(search_criteria):
  p_search_criteria = urllib.quote(search_criteria)
  req = requests.get("https://www.googleapis.com/youtube/v3/search?key={key}&fields=items(id(videoId))&part=id,snippet&q={query}&maxResults=50".format(query = p_search_criteria, key = api_key))
  retrieved_videos = eval(req.content)
  videos, tags = [], []
  for item in retrieved_videos['items']:
      videoid = item['id']['videoId']
      req = requests.get("https://www.googleapis.com/youtube/v3/videos?key={key}&fields=items(snippet(tags))&part=snippet&id={video_id}".format(video_id = videoid, key = api_key))
      try:
        tag = eval(req.content)['items'][0]['snippet']['tags']
      except:
        tag = []
      tags.append(list(tag))
      videos.append(videoid)
  return videos, tags


print("Expected " + str(len(search_criteria) * 50) + " videos")

vid_id, tags = [],[]
for to_search in search_criteria:
  videos, tag = collect_videos(to_search)
  vid_id.append(videos)
  tags.append(tag)
  break


#youtube-dl -o out https://www.youtube.com/watch?v=GF60Iuh643I