import streamlit as st
import pandas as pd
import nltk
import re
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
import os
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report



st.set_page_config(
    page_icon="tlogo.png",
    page_title="Sentiment Analysis app",
    layout="wide"
)


# Sidebar Menu
with st.sidebar:
    st.markdown("""
        <style>
            /* Sidebar Menu Container */
            .sidebar-menu {
                background-color: #1e1e2f; /* Dark Purple */
                padding: 15px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(255, 255, 255, 0.1);
            }

            /* Sidebar Title */
            .sidebar-title {
                text-align: center;
                font-size: 22px;
                font-weight: bold;
                color: white;
                padding-bottom: 10px;
            }
        </style>

        <div class="sidebar-menu">
            <h2 class='sidebar-title'>Menu</h2>
    """, unsafe_allow_html=True)

    menu = st.radio(
        "",
        ["🏠 Home", "ℹ️ About", "⭐ Reviews", "📞 Contact"]
    )

    st.markdown("</div>", unsafe_allow_html=True)  # Close menu div
    
    # Footer Section
    st.markdown("""
        <style>
            .footer {
                background-color: #f0f0f0; /* Light Gray */
                color: #333; /* Dark Text */
                text-align: center;
                padding: 10px;
                font-size: 16px;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                margin-top: 20px;
            }
        </style>

        <div class="footer">
            Made with ❤️ by Zia Ullah
        </div>
    """, unsafe_allow_html=True)


# Detect Light/Dark Mode
theme = st.get_option("theme.base")  
is_dark = theme == "dark"  

# Define Adaptive Colors
primary_color = "#15f4ee" if is_dark else "#007acc"  # Neon Cyan (Dark) | Blue (Light)
secondary_color = "#00f7ff" if is_dark else "#00509e"  # Brighter Cyan (Dark) | Deep Blue (Light)
text_color = "white" if is_dark else "black"  
bg_color = "#001f3f" if is_dark else "#f9f9f9" 


# Download required NLTK model
nltk.download("vader_lexicon")

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(tweet):
    """Analyze sentiment using VADER."""
    sentiment_score = sia.polarity_scores(tweet)
    if sentiment_score["compound"] >= 0.05:
        return "Positive"
    elif sentiment_score["compound"] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def Home():
    st.markdown("""
        <style>
            /* Title Styling */
            .title {
                text-align: left;
                font-size: 48px;
                font-weight: bold;
                margin-left: 20px;
                padding-top: 10px;
                transition: color 0.3s ease-in-out;
            }
        </style>

        <!-- Title (Color Will Change Dynamically) -->
        <h1 id="dynamic-title" class="title">TweetMood Analyzer</h1>

        <script>
            function updateTitleColor() {
                const theme = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
                document.getElementById("dynamic-title").style.color = theme === "dark" ? "white" : "black";
            }

            // Run on load
            updateTitleColor();

            // Listen for theme changes
            window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", updateTitleColor);
        </script>
    """, unsafe_allow_html=True)

    # Input Section
    tweet = st.text_area("Enter your tweet:", "", height=150, key="tweet_input")

    # Sentiment Analysis Button
    if st.button("Analyze Sentiment", key="analyze_button"):
        if tweet:
            result = analyze_sentiment(tweet)
            color = "#00ff00" if result == "Positive" else "#ff4d4d" if result == "Negative" else "#f4d03f"

            st.markdown(f"""
                <div class="sentiment-result" style="color: {color}; font-size: 20px; font-weight: bold; margin-top: 10px;">
                    🎯 Sentiment: <b>{result}</b>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter a tweet to analyze.")



#user Define Function for About
def About ():
    st.header("📝 About This Application")

    # App Description
    st.write(
        """
        Welcome to **Twitter Sentiment Analysis**, a cutting-edge tool that leverages AI and Natural Language Processing (NLP) 
        to analyze tweets and determine their sentiment—**Positive, Negative, or Neutral**.  
        This app helps businesses, researchers, and individuals gain **insights** into public opinion and trends.
        """
    )

    # Features Section
    st.subheader("🚀 Key Features")
    features = [
        "✅ **Real-time Sentiment Analysis** of Tweets",
        "✅ **AI-Powered Accuracy** using NLP models",
        "✅ **User-Friendly Interface** with instant results",
        "✅ **Secure & Fast Processing**",
        "✅ **Admin Panel** to manage reviews"
    ]
    for feature in features:
        st.write(feature)

    # How It Works
    st.subheader("🛠 How It Works?")
    st.write(
        """
        1️⃣ **Enter a Tweet**: Type or paste any tweet into the input box.  
        2️⃣ **Click 'Analyze Sentiment'**: The AI model will process the text.  
        3️⃣ **Get Instant Results**: The system classifies the tweet as **Positive, Negative, or Neutral**.  
        """
    )

    # Tech Stack
    st.subheader("💡 Technologies Used")
    st.write(
        """
        - 🐍 **Python** (Powering AI and Machine Learning)  
        - 🤖 **Natural Language Processing (NLP)** for sentiment analysis  
        - 📊 **Streamlit** for an interactive user interface  
        - 🗄 **SQLite** for database management  
        """
    )

    # About the Creator
    st.subheader("👨‍💻 About the Developer")
    st.write(
        """
        **Zia Ullah** is a passionate Software Engineer specializing in **Machine Learning, Deep Learning, and Data Analysis**.  
        He developed this application to demonstrate the power of AI in understanding human emotions through text.  
        📌 **Connect with him on [LinkedIn](https://www.linkedin.com/in/engr-ziaullah-7672ab260)**.
        """
    )

    # Closing Statement
    st.info("📢 Have feedback? Head over to the **Reviews** section and let us know what you think!")




#this is also a user define Function for reviews
def Reviews():
    st.header("📝 User Reviews")

    # Connect to Database
    conn = sqlite3.connect("reviews.db", check_same_thread=False)
    cursor = conn.cursor()

    # Create Reviews Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            review TEXT
        )
    """)
    conn.commit()

    # Function to Validate Email
    def is_valid_email(email):
        pattern = r'^[\w\.-]+@gmail\.com$'
        return re.match(pattern, email) is not None

    # Review Submission Form
    with st.form("review_form"):
        name = st.text_input("Enter your Name")
        email = st.text_input("Enter your Email")
        review = st.text_area("Write your Review")
        submitted = st.form_submit_button("Submit Review")

        if submitted:
            if not name or not email or not review:
                st.error("❌ Please fill out all fields.")
            elif not is_valid_email(email):
                st.error("❌ Please enter a valid email address ending with @gmail.com.")
            else:
                cursor.execute("INSERT INTO reviews (name, email, review) VALUES (?, ?, ?)", (name, email, review))
                conn.commit()
                st.success("✅ Your review has been submitted successfully!")

    # Buttons for Reviews & Admin Panel
    col1, col2 = st.columns(2)

    with col1:
        if st.button("📢 Show All Reviews"):
            st.session_state.show_reviews = True

    # Show All Reviews
    if "show_reviews" in st.session_state and st.session_state.show_reviews:
        st.subheader("📋 All User Reviews")
        cursor.execute("SELECT name, review FROM reviews")
        reviews = cursor.fetchall()

        if not reviews:
            st.info("ℹ️ No reviews found.")
        else:
            for review in reviews:
                st.write(f"**🧑 {review[0]}:** {review[1]}")

        # Show Admin Panel button only after displaying reviews
        if st.button("🔐 Admin Panel"):
            st.session_state.show_admin_login = True

    # Admin Login Page (Appears Only After Clicking "Admin Panel")
    if "show_admin_login" in st.session_state and st.session_state.show_admin_login:
        st.subheader("🔑 Admin Login")
        admin_email = st.text_input("📧 Admin Email")
        admin_password = st.text_input("🔒 Admin Password", type="password")
        admin_login = st.button("Login as Admin")

        if admin_login:
            if admin_email == "admin@gmail.com" and admin_password == "admin@##123":
                st.session_state.admin_logged_in = True
                st.success("✅ Admin Logged In Successfully!")

        # Admin Panel (Appears Only If Logged In)
        if "admin_logged_in" in st.session_state and st.session_state.admin_logged_in:
            st.subheader("🛠 Manage Reviews")

            # Fetch and Display All Reviews for Admin
            cursor.execute("SELECT * FROM reviews")
            reviews = cursor.fetchall()

            if not reviews:
                st.info("ℹ️ No reviews found.")
            else:
                review_ids_to_delete = []
                for review in reviews:
                    st.write(f"**🆔 ID:** {review[0]}")
                    st.write(f"**🧑 Name:** {review[1]}")
                    st.write(f"**📧 Email:** {review[2]}")
                    st.write(f"**💬 Review:** {review[3]}")

                    # Use a form for delete buttons to handle state properly
                    with st.form(f"delete_form_{review[0]}"):
                        delete_button = st.form_submit_button(f"🗑 Delete Review {review[0]}")

                        if delete_button:
                            review_ids_to_delete.append(review[0])

                # Process Deletion
                if review_ids_to_delete:
                    for review_id in review_ids_to_delete:
                        cursor.execute("DELETE FROM reviews WHERE id = ?", (review_id,))
                    conn.commit()
                    st.success("✅ Selected review(s) deleted! Refreshing...")
                    st.experimental_rerun()




#This a user define function for Contact from
def Contact ():
    st.subheader("Contact")
    st.header(":mailbox: Get In Touch With Us!")

    contact_form = """
    <form action="https://formsubmit.co/ziaullahbj9@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your email" required>
        <textarea name="message" placeholder="Your message here"></textarea>
        <button type="submit">Send</button>
    </form>
    """
    st.markdown(contact_form, unsafe_allow_html=True)

    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    local_css("contact_form.css")

#TThe main function.
if menu == "🏠 Home":
    Home()
    
elif menu == "ℹ️ About":
    About()
    
elif menu == "⭐ Reviews":
    Reviews()

elif menu == "📞 Contact":
    Contact()


