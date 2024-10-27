import os

# Make sure this is the correct path to your `app.py`
app_path = r"C:\Users\Rajiv\Downloads\MRU\Trading_App\app.py"

def launch_app():
    # Launch the Streamlit app
    print(f"Launching Streamlit app: {app_path}")
    os.system(f"streamlit run {app_path}")

if __name__ == "__main__":
    launch_app()
