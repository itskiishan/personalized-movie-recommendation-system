import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Personalized Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


# --------------------------------------------------
# Load Deployment Data
# --------------------------------------------------

@st.cache_resource
def load_model_data():

    base_path = "model/deployment"

    with open(
        f"{base_path}/sparse_feature_matrix.pkl",
        "rb"
    ) as file:
        sparse_feature_matrix = pickle.load(file)

    with open(
        f"{base_path}/user_ids.pkl",
        "rb"
    ) as file:
        user_ids = pickle.load(file)

    with open(
        f"{base_path}/movie_ids.pkl",
        "rb"
    ) as file:
        movie_ids = pickle.load(file)

    with open(
        f"{base_path}/user_clusters.pkl",
        "rb"
    ) as file:
        user_cluster_data = pickle.load(file)

    with open(
        f"{base_path}/movie_features.pkl",
        "rb"
    ) as file:
        movie_features = pickle.load(file)

    return (
        sparse_feature_matrix,
        user_ids,
        movie_ids,
        user_cluster_data,
        movie_features
    )


(
    sparse_feature_matrix,
    user_ids,
    movie_ids,
    user_cluster_data,
    movie_features
) = load_model_data()


# --------------------------------------------------
# Recommendation Function
# --------------------------------------------------

def recommend_movies_clustered(
    user_id,
    num_recommendations=10,
    num_similar_users=20
):

    # Find target user's index
    user_indices = np.where(user_ids == user_id)[0]

    if len(user_indices) == 0:
        raise ValueError(
            f"User ID {user_id} does not exist."
        )

    user_index = user_indices[0]


    # Get target user's cluster
    cluster_row = user_cluster_data[
        user_cluster_data["UserID"] == user_id
    ]

    if cluster_row.empty:
        raise ValueError(
            f"Cluster information not found for User {user_id}."
        )

    target_cluster = cluster_row["Cluster"].iloc[0]


    # Get users in the same cluster
    cluster_users = user_cluster_data[
        user_cluster_data["Cluster"] == target_cluster
    ]["UserID"].to_numpy()


    # Convert cluster user IDs to matrix indices
    cluster_indices = np.array([
        np.where(user_ids == uid)[0][0]
        for uid in cluster_users
    ])


    # Extract users belonging to the target cluster
    cluster_matrix = sparse_feature_matrix[
        cluster_indices
    ]


    # Calculate similarity of target user
    # with users in the same cluster
    similarity_scores = cosine_similarity(
        sparse_feature_matrix[user_index],
        cluster_matrix
    ).flatten()


    # Remove target user
    valid_mask = cluster_users != user_id

    cluster_users = cluster_users[valid_mask]
    similarity_scores = similarity_scores[valid_mask]


    # Select most similar users
    top_indices = np.argsort(
        similarity_scores
    )[::-1][:num_similar_users]


    similar_user_indices = cluster_indices[
        valid_mask
    ][top_indices]

    similar_scores = similarity_scores[
        top_indices
    ]


    # Get ratings from similar users
    similar_users_ratings = sparse_feature_matrix[
        similar_user_indices
    ]


    # Calculate weighted recommendation scores
    weighted_scores = (
        similar_scores @ similar_users_ratings
    )

    weighted_scores = np.asarray(
        weighted_scores
    ).ravel()


    # Get target user's ratings
    user_ratings = (
        sparse_feature_matrix[user_index]
        .toarray()
        .ravel()
    )


    # Do not recommend already-rated movies
    weighted_scores[user_ratings > 0] = -1


    # Get top movie indices
    recommended_indices = np.argsort(
        weighted_scores
    )[::-1][:num_recommendations]


    # Get movie IDs
    recommended_movie_ids = movie_ids[
        recommended_indices
    ]


    # Create recommendation dataframe
    recommendations = movie_features[
        movie_features["MovieID"].isin(
            recommended_movie_ids
        )
    ].copy()


    # Preserve recommendation order
    recommendation_order = {
        movie_id: rank + 1
        for rank, movie_id
        in enumerate(recommended_movie_ids)
    }


    score_mapping = dict(
        zip(
            recommended_movie_ids,
            weighted_scores[recommended_indices]
        )
    )


    recommendations["Rank"] = (
        recommendations["MovieID"]
        .map(recommendation_order)
    )


    recommendations["Score"] = (
        recommendations["MovieID"]
        .map(score_mapping)
    )


    recommendations = recommendations.sort_values(
        "Rank"
    )


    return (
        recommendations.reset_index(drop=True),
        target_cluster
    )


# --------------------------------------------------
# Streamlit Interface
# --------------------------------------------------

st.title(
    "🎬 Personalized Movie Recommendation System"
)

st.write(
    "A collaborative filtering based movie "
    "recommendation system using K-Means clustering "
    "and cosine similarity."
)

st.divider()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("Recommendation Settings")

user_id = st.sidebar.number_input(
    "Enter User ID",
    min_value=1,
    max_value=6040,
    value=1,
    step=1
)

num_recommendations = st.sidebar.slider(
    "Number of Recommendations",
    min_value=5,
    max_value=20,
    value=10
)

num_similar_users = st.sidebar.slider(
    "Number of Similar Users",
    min_value=5,
    max_value=50,
    value=20
)


# --------------------------------------------------
# Generate Recommendations
# --------------------------------------------------

if st.button(
    "🎯 Get Recommendations",
    type="primary"
):

    try:

        recommendations, cluster = (
            recommend_movies_clustered(
                user_id=user_id,
                num_recommendations=num_recommendations,
                num_similar_users=num_similar_users
            )
        )

        st.success(
            f"Recommendations generated for User {user_id}"
        )

        st.info(
            f"User {user_id} belongs to Cluster {cluster}."
        )

        st.subheader("Recommended Movies")

        display_df = recommendations[
            [
                "Rank",
                "Movie_Title",
                "Genres",
                "Score"
            ]
        ].copy()

        display_df["Score"] = (
            display_df["Score"].round(2)
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:

        st.error(
            f"Unable to generate recommendations: {e}"
        )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Movie recommendations are generated using "
    "K-Means user clustering and collaborative "
    "filtering with cosine similarity."
)