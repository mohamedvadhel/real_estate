import streamlit as st

def main():
    # Set the page title and description
    st.set_page_config(page_title='Hello World', page_icon="🌍")

    # Add the title to the interface
    st.title('Hello, World!')

    # Write a message
    st.write('Welcome to this simple Streamlit app that says "Hello, World!"')

if __name__ == '__main__':
    main()
