import os
from dotenv import load_dotenv
import json

import pandas as pd
import spotipy
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

    # Current User Display Nmae
    print('User Display Name')
    print('--------------------')
    print(sp.current_user()['display_name'])
    print('--------------------')
    
    # Current User Image
    print('User Image')
    print('--------------------')
    print(sp.current_user()['images'][0]['url'])
    print('--------------------')


    # Short Term Top Artists
    print('Recent Top Artists')
    print('--------------------')
    user_top_artists_results = sp.current_user_top_artists(time_range='short_term', limit=10)
    for i, item in enumerate(user_top_artists_results['items']):
        print(i, item['name'])
    print('--------------------')

    # Short Term Top Tracks
    print('Recent Top Tracks')
    print('------------------')

    user_top_tracks_results = sp.current_user_top_tracks(time_range='short_term', limit=10)   
    for i, item, in enumerate(user_top_tracks_results['items']):
        print(i, item['name'] +  ' by ' + item['album']['artists'][0]['name'] + ' (' + item['uri'] + ') ')
    print('--------------------')


if __name__ == '__main__':
    main()
