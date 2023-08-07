import streamlit as st

def main():
    # Set the page title and description
    st.set_page_config(page_title='Khadouj El Yedali', page_icon=":heart:")

    # Add a personalized message to Khadouj
    st.title('Hello, Khadouj!')
    st.write(" 4i interface application s9ira mou3adla lla lk nti")

    # Upload the video file
    uploaded_file = st.file_uploader("Upload your video", type=["mp4", "avi", "mov"])

    # Check if a video file was uploaded
    if uploaded_file is not None:
        # Display the video
        st.video(uploaded_file)

if __name__ == '__main__':
    main()
