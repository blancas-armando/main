import os
from dotenv import load_dotenv

import pandas as pd
import spotipy
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
import streamlit as st


def main():
    load_dotenv()
    cid = os.getenv('SPOTIFY_CLIENT_ID')
    secret = os.getenv('SPOTIFY_CLIENT_SECRET')
    client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
    spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    artist_name = 'Radiohead'
    results = spotify.search(q='artist:' + artist_name, type='artist')
    items = results['artists']['items']
    if len(items) > 0:
        artist = items[0]
        print(artist['name'], artist['images'][0]['url'])

    st.title('Artist Name and Photo')
    st.image(artist['images'][0]['url'])
    st.caption(artist['name'])

if __name__ == '__main__':
    main()
