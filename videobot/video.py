import streamlit as st

# Set the title of the Streamlit app
st.title("User Prompt Input App")

# Create a text input widget for the user to enter a prompt
user_input = st.text_input("Enter your prompt:")

# Display the user's input back to them
if user_input:
    st.subheader("You entered:")
    st.write(user_input)

# Optionally, you can add more features or processing based on the user's input
