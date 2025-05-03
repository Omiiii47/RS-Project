from flask import Flask, render_template,request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np

# app
app = Flask(__name__)

# Load datasets and models
features = ['Name_x', 'State', 'Type', 'BestTimeToVisit', 'Preferences', 'Gender', 'NumberOfAdults', 'NumberOfChildren']
model = pickle.load(open('model.pkl','rb'))
label_encoders = pickle.load(open('label_encoders.pkl','rb'))

destinations_df = pd.read_csv("Expanded_Destinations.csv")
userhistory_df = pd.read_csv("Final_Updated_Expanded_UserHistory.csv")
df = pd.read_csv("final_df.csv")


# Collaborative Filtering Function
# Create a user-item matrix based on user history
user_item_matrix = userhistory_df.pivot(index='UserID', columns='DestinationID', values='ExperienceRating')

# Fill missing values with 0 (indicating no rating/experience)
user_item_matrix.fillna(0, inplace=True)

# Compute cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix)


# Function to recommend destinations based on user similarity
def collaborative_recommend(user_id, user_similarity, user_item_matrix, destinations_df):
    """
    Recommends destinations based on collaborative filtering.

    Args:
    - user_id: ID of the user for whom recommendations are to be made.
    - user_similarity: Cosine similarity matrix for users.
    - user_item_matrix: User-item interaction matrix (e.g., ratings or preferences).
    - destinations_df: DataFrame containing destination details.

    Returns:
    - DataFrame with recommended destinations and their details.
    """
    # Find similar users
    similar_users = user_similarity[user_id - 1]

    # Get the top 5 most similar users
    similar_users_idx = np.argsort(similar_users)[::-1][1:6]

    # Get the destinations liked by similar users
    similar_user_ratings = user_item_matrix.iloc[similar_users_idx].mean(axis=0)

    # Recommend the top 5 destinations
    recommended_destinations_ids = similar_user_ratings.sort_values(ascending=False).head(5).index

    # Filter the destinations DataFrame to include detailed information
    recommendations = destinations_df[destinations_df['DestinationID'].isin(recommended_destinations_ids)][[
        'DestinationID', 'Name', 'State', 'Type', 'Popularity', 'BestTimeToVisit'
    ]]

    return recommendations

# Prediction system
def recommend_destinations(user_input, model, label_encoders, features, data):
    # Encode user input
    encoded_input = {}
    for feature in features:
        if feature in label_encoders:
            encoded_input[feature] = label_encoders[feature].transform([user_input[feature]])[0]
        else:
            encoded_input[feature] = user_input[feature]

    # Convert to DataFrame
    input_df = pd.DataFrame([encoded_input])

    # Predict popularity
    predicted_popularity = model.predict(input_df)[0]

    return predicted_popularity


# Route for the Home Page
@app.route('/')
def index():
    return render_template('index.html')
# Route for Travel Recommendation Page
@app.route('/recommendation')
def recommendation():
    return render_template('recommendation.html')

# Route for the recommendation
@app.route("/recommend", methods=['GET', 'POST'])
def recommend():
    if request.method == "POST":
        try:
            # Only validate required fields
            num_adults = int(request.form['adults'])
            if num_adults < 1 or num_adults > 10:
                return render_template('recommendation.html', 
                    error="Number of adults must be between 1 and 10")
            
            # Make children optional with default value 0
            num_children = int(request.form.get('children', 0))
            if num_children < 0 or num_children > 10:
                return render_template('recommendation.html', 
                    error="Number of children must be between 0 and 10")

            user_id = int(request.form['user_id'])
            
            # Create user_input with default values for optional fields
            user_input = {
                'Name_x': request.form.get('name', ''),
                'Type': request.form.get('type', ''),
                'State': request.form.get('state', ''),
                'BestTimeToVisit': request.form.get('best_time', ''),
                'Preferences': request.form.get('preferences', ''),
                'Gender': request.form.get('gender', ''),
                'NumberOfAdults': num_adults,
                'NumberOfChildren': num_children,
            }

            # If optional field is empty, use most popular value from training data
            for feature in features:
                if not user_input[feature] and feature in label_encoders:
                    # Use most common value from training data
                    user_input[feature] = df[feature].mode()[0]

            recommended_destinations = collaborative_recommend(user_id, user_similarity,
                                                           user_item_matrix, destinations_df)
            predicted_popularity = recommend_destinations(user_input, model, label_encoders, features, df)

            return render_template('recommendation.html', 
                                recommended_destinations=recommended_destinations,
                                predicted_popularity=predicted_popularity)
                                
        except ValueError:
            return render_template('recommendation.html', 
                error="Please enter valid numbers for adults and children")
            
    return render_template('recommendation.html')


if __name__ == '__main__':
    # Run the app in debug mode
    app.run(debug=True)
