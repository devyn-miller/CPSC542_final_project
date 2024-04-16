# installment requirements: pip install pytube

from pytube import YouTube, Playlist
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def download_video(video_url, resolution='720p'): 
    '''
    download a video from youtube

    :param video_url: the url of the video 
    :type video_url: str
    :param resolution: the resolution of the video, defaults to '720p'
    :type resolution: str, optional
    '''
    youtube = YouTube(video_url)
    youtube = youtube.streams.get_by_resolution(resolution)
    try: 
        youtube.download()
    except: 
        print("an error has occured")
    print("download successful")
    

def download_playlist(playlist_url, resolution='720p'): 
    '''
    download all videos in a playlist

    :param playlist_url: the url of the playlist
    :type playlist_url: str
    :param resolution: the resolution of the video, defaults to '720p'
    :type resolution: str, optional
    '''
    playlist = Playlist(playlist_url)
    for video_url in playlist.video_urls: 
        download_video(video_url, resolution)
        

download_playlist('https://youtube.com/playlist?list=PLfbJcGXZCB6wYbR98BMNQKlXb28GuIQEG&si=6Gquu4vlwLLEQV2o')