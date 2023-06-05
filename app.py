import streamlit as st
import hashlib

# Create a login form
st.title("User Login")

username = st.text_input("Username", key="username")
password1 = st.text_input("Password", type="password", key="password1")

# If the user enters a valid username and password, log them in
if st.button("Login"):
    if username == "admin" and password1 == "password":
        st.success("Login successful!")
        st.redirect("/recommendation")
    else:
        st.warning("Invalid username or password.")

# Create a register form
st.write("Enter your details below to create a new account.")

first_name = st.text_input("First name", key="first_name")
last_name = st.text_input("Last name", key="last_name")
email = st.text_input("Email", key="email")
password2 = st.text_input("Password", type="password", key="password2")

# If the user enters all of the required details, create the account
if st.button("Create Account"):
    if first_name and last_name and email and password2:
        # Hash the password
        hashed_password = hashlib.sha256(password2.encode()).hexdigest()

        # Create a new user in the database
        user_data = {
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "password": hashed_password,
        }
        st.database.insert(user_data)

        st.success("Account created successfully!")
        st.redirect("/recommendation")
    else:
        st.warning("Please enter all of the required details.")

# Redirect the user to the recommendation page
st.stop()



st.set_page_config(page_title="Music Recommendation", layout="wide")

import pandas as pd
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import streamlit.components.v1 as components

@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv("data/processed_track_df.csv")
    df['genres'] = df.genres.apply(lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")])
    exploded_track_df = df.explode("genres")
    return exploded_track_df

genre_names = ['Dance Pop', 'Electronic', 'Electropop', 'Hip Hop', 'Jazz', 'K-pop', 'Latin', 'Pop', 'Pop Rap', 'R&B', 'Rock']
audio_feats = ["acousticness", "danceability", "energy", "instrumentalness", "valence", "tempo"]

exploded_track_df = load_data()

def n_neighbors_uri_audio(genre, start_year, end_year, test_feat):
    genre = genre.lower()
    genre_data = exploded_track_df[(exploded_track_df["genres"]==genre) & (exploded_track_df["release_year"]>=start_year) & (exploded_track_df["release_year"]<=end_year)]
    genre_data = genre_data.sort_values(by='popularity', ascending=False)[:500]

    neigh = NearestNeighbors()
    neigh.fit(genre_data[audio_feats].to_numpy())

    n_neighbors = neigh.kneighbors([test_feat], n_neighbors=len(genre_data), return_distance=False)[0]

    uris = genre_data.iloc[n_neighbors]["uri"].tolist()
    audios = genre_data.iloc[n_neighbors][audio_feats].to_numpy()
    return uris, audios

def page():
    title = "Music Recommendation System"
    st.title(title)

    st.write("Welcome! Here you can listen to the songs recommended by our system and customize it based on audio features and genres!")
    st.markdown("##")

    with st.container():
        col1, col2,col3,col4 = st.columns((2,0.5,0.5,0.5))
        with col3:
            st.sidebar.header("***Select genre:***")
            genre = st.sidebar.radio(
                "",
                genre_names, index=genre_names.index("K-pop"))
        with col1:
            st.markdown("***Feature customization:***")
            start_year, end_year = st.slider(
                'Year Range',
                1990, 2023, (2010, 2023)
            )
            acousticness = st.slider(
                'Acousticness',
                0.0, 1.0, 0.5)
            danceability = st.slider(
                'Danceability',
                0.0, 1.0, 0.5)
            energy = st.slider(
                'Energy',
                0.0, 1.0, 0.5)
            instrumentalness = st.slider(
                'Instrumentalness',
                0.0, 1.0, 0.0)
            valence = st.slider(
                'Valence',
                0.0, 1.0, 0.45)
            tempo = st.slider(
                'Tempo',
                0.0, 244.0, 118.0)

    tracks_per_page = 8
    test_feat = [acousticness, danceability, energy, instrumentalness, valence, tempo]
    uris, audios = n_neighbors_uri_audio(genre, start_year, end_year, test_feat)

    tracks = []
    for uri in uris:
        track = """<iframe src="https://open.spotify.com/embed/track/{}" width="260" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>""".format(uri)
        tracks.append(track)

    if 'previous_inputs' not in st.session_state:
        st.session_state['previous_inputs'] = [genre, start_year, end_year] + test_feat
    
    current_inputs = [genre, start_year, end_year] + test_feat
    if current_inputs != st.session_state['previous_inputs']:
        if 'start_track_i' in st.session_state:
            st.session_state['start_track_i'] = 0
        st.session_state['previous_inputs'] = current_inputs

    if 'start_track_i' not in st.session_state:
        st.session_state['start_track_i'] = 0
    
    with st.container():
        col1, col2, col3 = st.columns([2,1,2])
        if st.button("Recommend More Songs"):
            if st.session_state['start_track_i'] < len(tracks):
                st.session_state['start_track_i'] += tracks_per_page

        current_tracks = tracks[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
        current_audios = audios[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
        if st.session_state['start_track_i'] < len(tracks):
            for i, (track, audio) in enumerate(zip(current_tracks, current_audios)):
                if i%2==0:
                    with col1:
                        components.html(
                            track,
                            height=400,
                        )
                        with st.expander("See details"):
                            df = pd.DataFrame(dict(
                            r=audio[:5],
                            theta=audio_feats[:5]))
                            fig = px.line_polar(df, r='r', theta='theta', line_close=True)
                            fig.update_layout(height=400, width=340)
                            st.plotly_chart(fig)
            
                else:
                    with col3:
                        components.html(
                            track,
                            height=400,
                        )
                        with st.expander("See details"):
                            df = pd.DataFrame(dict(
                                r=audio[:5],
                                theta=audio_feats[:5]))
                            fig = px.line_polar(df, r='r', theta='theta', line_close=True)
                            fig.update_layout(height=400, width=340)
                            st.plotly_chart(fig)

        else:
            st.write("No songs left to recommend")

page()
