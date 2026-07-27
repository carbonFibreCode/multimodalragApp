def get_custom_css():
    """Generate custom CSS for Streamlit app."""
    return """
    <style>
        /* Hide Streamlit header and footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* General container styling */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 3rem;
            padding-right: 3rem;
        }

        /* Improve widget styling */
        .st-emotion-cache-1y4p8pa { /* File uploader container */
            border: 1px dashed #ced4da;
            border-radius: 0.5rem;
        }
        
        .stButton>button {
            border-radius: 0.5rem;
            width: 100%;
        }

        /* Chat message styling */
        .stChatMessage {
            border-radius: 0.75rem;
            border: 1px solid #e9ecef;
            padding: 1rem;
        }
    </style>
    """