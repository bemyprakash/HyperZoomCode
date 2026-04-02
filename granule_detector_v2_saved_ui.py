import streamlit as st
import os

# Function to get folder path from user input
def get_folder_path():
    folder_path = st.text_input('Enter the folder path:')
    return folder_path if os.path.isdir(folder_path) else None

# Main function for the granule detector 
def main():
    st.title('Granule Detector v2')
    path = get_folder_path()
    if path:
        st.success('Folder path received: {}'.format(path))
        # Add your granule detection logic here
    else:
        st.error('Invalid folder path! Please try again.')

if __name__ == '__main__':
    main()