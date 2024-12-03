import streamlit as st
import librosa
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.metrics.pairwise import cosine_similarity

# Feature extraction function
def extract_audio_features(audio_file):
    try:
        y, sr = librosa.load(audio_file, sr=None)
        features = {}
        
        # Extract MFCCs and flatten
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfcc, axis=1)
        for i, coeff in enumerate(mfcc_mean):
            features[f'mfcc_{i+1}'] = coeff
            
        # Extract tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = float(tempo)
        
        # Extract other scalar features with error handling
        features['spectral_rolloff'] = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        features['chroma'] = float(np.mean(librosa.feature.chroma_stft(y=y, sr=sr)))
        features['spectral_centroid'] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        features['spectral_flatness'] = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        
        return features
    except Exception as e:
        st.error(f"Error processing audio file: {str(e)}")
        return None

def load_models():
    try:
        kmeans = joblib.load('kmeans_model.joblib')
        scaler = joblib.load('scaler.joblib')
        train_df = pd.read_csv('music_features.csv')
        if kmeans is None or scaler is None or train_df.empty:
            raise ValueError("One or more required models failed to load")
        return kmeans, scaler, train_df
    except FileNotFoundError as e:
        st.error(f"Required model files not found: {str(e)}")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

def get_recommendations(extracted_features, kmeans, scaler, train_df):
    # Convert features to correct format
    feature_vector = np.array(list(extracted_features.values())).reshape(1, -1)
    
    # Get training features
    train_features = train_df.drop(['song_name', 'cluster'], axis=1)
    
    # Scale features
    scaled_features = scaler.transform(feature_vector)
    scaled_train_features = scaler.transform(train_features)
    
    # Predict cluster
    predicted_cluster = kmeans.predict(scaled_features)[0]
    
    # Get songs in the same cluster
    cluster_songs = train_df[train_df['cluster'] == predicted_cluster].copy()
    cluster_features = scaled_train_features[train_df['cluster'] == predicted_cluster]
    
    # Calculate similarities
    similarities = cosine_similarity(scaled_features, cluster_features)[0]
    cluster_songs['similarity'] = similarities * 100
    
    return predicted_cluster, cluster_songs.sort_values('similarity', ascending=False)

def main():
    st.title("Music Recommendation System")
    st.write("Upload a music file to get personalized recommendations!")
    
    # Load models
    kmeans, scaler, train_df = load_models()
    
    # Check if any of the required components failed to load
    if kmeans is None or scaler is None or train_df is None:
        st.error("Failed to initialize the recommendation system. Please check the model files.")
        st.stop()
        return
    
    # File upload
    uploaded_file = st.file_uploader("Upload a music file", type=["mp3", "wav"])
    
    if uploaded_file is not None:
        # Display audio player
        st.audio(uploaded_file, format="audio/mp3")
        
        # Process the file
        with st.spinner("Processing audio file..."):
            # Save temporary file
            temp_path = "temp_audio_file"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Extract features
            extracted_features = extract_audio_features(temp_path)
            
            # Clean up temporary file
            try:
                os.remove(temp_path)
            except Exception as e:
                st.warning(f"Could not remove temporary file: {str(e)}")
            
            if extracted_features is not None:
                # Get recommendations
                predicted_cluster, recommendations = get_recommendations(
                    extracted_features, kmeans, scaler, train_df
                )
                
                # Display results
                st.subheader("Top 5 Recommended Songs:")
                for idx, row in recommendations.head(5).iterrows():
                    st.write(
                        f"🎵 {row['song_name']} "
                        f"(Similarity: {row['similarity']:.1f}%)"
                    )

if __name__ == "__main__":
    main()