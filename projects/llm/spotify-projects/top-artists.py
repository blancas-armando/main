import os
from dotenv import load_dotenv

import pandas as pd
import spotipy
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth
import streamlit as st


def main():
    load_dotenv()
    cid = os.getenv('SPOTIFY_CLIENT_ID')
    secret = os.getenv('SPOTIFY_CLIENT_SECRET')
    scope = 'user-top-read'
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=cid,
                                                   client_secret=secret,
                                                   redirect_uri='https://flippingbook.com/404',
                                                   scope=scope))

    # Short Term Top Artists
    print('Recent Top Artists')
    print('--------------------')
    user_top_artists_results = sp.current_user_top_artists(time_range='short_term', limit=1)
    for i, item in enumerate(user_top_artists_results['items']):
        print(i, item['name'])
    print('--------------------')

    # Short Term Top Tracks
    print('Recent Top Tracks')
    print('------------------')
    user_top_tracks_results = sp.current_user_top_tracks(time_range='short_term', limit=1)
    print(user_top_tracks_results)
    for i, item in enumerate(user_top_tracks_results['items']):
        print(i, item['name'])
    print('--------------------')

if __name__ == '__main__':
    main()
