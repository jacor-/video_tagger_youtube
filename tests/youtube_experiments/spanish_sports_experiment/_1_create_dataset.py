from datasets.dataset_creators.YoutubeDataserCreator import YoutubeVideoCollector
import settings as st

videos_per_search = 500
frames_per_video = 20

api_key = st.youtube_api_key
search_criteria = [#"dog", "cat", "bird", "turtle",
                   #"fight", "run", "walk", "climb",
                   "messi neymar suarez", "ronado benzema bale", "guardiola mourinho simeone","barca chelsea munich",
                   "guardiola", "mourinho","lebron james", "marc gasol", "pau gasol", "tiger woods", "simeone",
                   #"barcelona", "london", "rome", "amsterdam", "europe", "world",
                   #"plane", "car", "bike",
                   #"mariano rajoy", "pablo iglesias", "pedro sanchez", "albert rivera"
                   ]

youtube_videos = YoutubeVideoCollector(api_key, "youtube_experiment", search_criteria = search_criteria, videos_per_search = videos_per_search, frames_per_video = frames_per_video)
