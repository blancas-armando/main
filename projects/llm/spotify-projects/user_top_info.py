import os
from dotenv import load_dotenv
import json

import pandas as pd
import numpy as np
import urllib.request
import cv2

import spotipy
from spotipy.oauth2 import SpotifyOAuth
import streamlit as st
from PIL import Image


def main():
    load_dotenv()
    cid = os.getenv('SPOTIFY_CLIENT_ID')
    secret = os.getenv('SPOTIFY_CLIENT_SECRET')
    scope = 'user-top-read'
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=cid,
                                                   client_secret=secret,
                                                   redirect_uri='https://flippingbook.com/404',
                                                   scope=scope))

    # Current User Display Name and Image
    print('User Display Name')
    print('--------------------')
    display_name = sp.current_user()['display_name']
    print('--------------------')
    
    print('User Image')
    print('--------------------')
    print(sp.current_user()['images'][0]['url'])
    #TODO Need to find better way to convert and store user image; this works for now
    image = urllib.request.urlretrieve(url=sp.current_user()['images'][0]['url'], filename='spotify_user_image.png')
    print('--------------------')

    # with st.sidebar:
    #     st.title('Spotify User Analytics v1')
    #     st.image(Image.open('spotify_user_image.png'))
    #     st.text(f'{display_name}')

    # Short Term Top Artists
    print('Recent Top Artists')
    st.text('Recent Top Artists')
    print('--------------------')
    user_top_artists_results = sp.current_user_top_artists(time_range='short_term', limit=10)
    for i, item in enumerate(user_top_artists_results['items']):
        #TODO Let's append these to a list and display list?
        print(i, item['name'])
        for i, genre in enumerate(item['genres']):
            print('Genre ' + str(i) + ': ' + genre)
        print('--------------------')
    print('--------------------')

    # Short Term Top Tracks
    print('Recent Top Tracks')
    st.text('Recent Top Tracks')
    print('------------------')
    user_top_tracks_results = sp.current_user_top_tracks(time_range='short_term', limit=10)
    for i, item, in enumerate(user_top_tracks_results['items']):
        print(i, item['name'] +  ' by ' + item['album']['artists'][0]['name'])
        track_features = sp.audio_features(tracks=item['uri'])
        for track in track_features:
            print('Danceability: ' + str(round(track['danceability'], 2)*100) + '%')
            print('Energy percentage: ' + str(round(track['energy'], 2)*100) + '%')
            print('Loudness: ' + str(track['loudness']))
            print('Tempo: ' + str(track['tempo']) + ' bpm')
            print('--------------------')
    print('--------------------')


if __name__ == '__main__':
    main()
