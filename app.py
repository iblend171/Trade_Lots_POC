from flask import Flask, render_template, request
import pandas as pd
from datetime import datetime
import warnings
from darvas import darvas
import webbrowser
import threading
from threading import Timer

from supabase import create_client, Client

# Ignore all warnings
warnings.simplefilter("ignore")

app = Flask(__name__)

# Call the Darvas method to get data & Run the function without waiting for its return
thread = threading.Thread(target=darvas)
thread.start()
# df_merged = dataframe_call[0]
# df_hist = dataframe_call[1]


# Call the Supabase db
supabaseUrl = 'https://qgfzhdbljfzpnlccntlp.supabase.co'
supabaseKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFnZnpoZGJsamZ6cG5sY2NudGxwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzIyNTI2NjUsImV4cCI6MjA0NzgyODY2NX0.09QOxaLRCY85Grk2yiyiFoD460zsREjNxUHnUdeU9qg'

# url: str = os.environ.get(supabaseUrl)
# key: str = os.environ.get(supabaseKey)
supabase: Client = create_client(supabaseUrl, supabaseKey)


# Sign in with email and password
response = supabase.auth.sign_in_with_password(
    {"email": "iblend171@gmail.com", "password": "Tex@co!1"}
)

# Check the response
# if response.user:
#     print("Signed in successfully!")
#     print("User details:", response.user)
# else:
#     print("Sign-in failed:", response)

batch_size = 10
offset = 0
all_data = []

start_date = '2024-11-20'
end_date = '2026-01-01'

while True:
    response = supabase.table("TSX Darvas 2").select("*").gte("Close Date", start_date).lte("Close Date", end_date).limit(batch_size).offset(offset).execute()
    
    if not response.data:  # Break if no more data
        break

    all_data.extend(response.data)
    offset += batch_size

# Convert all data to a DataFrame
df_merged = pd.DataFrame(all_data)
print("Full data as DataFrame:")
print(df_merged)




# Function to style rows based on Trade column
def style_row(trade):
    if trade == "Long":
        return 'style="background-color: #d4edda; color: #155724;"'  # Green for Long
    elif trade == "Short":
        return 'style="background-color: #f8d7da; color: #721c24;"'  # Red for Short
    return ""

def apply_row_styles(df):
    styled_html = '<table class="table table-striped table-bordered">\n<thead>\n<tr>\n'

    # Add table headers
    for col in df.columns:
        styled_html += f'<th>{col}</th>\n'
    styled_html += '</tr>\n</thead>\n<tbody>\n'

    # Add table rows with styles
    for _, row in df.iterrows():
        row_style = style_row(row['Trade'])  # Get row style based on Trade value
        styled_html += f'<tr {row_style}>\n'
        for value in row:
            styled_html += f'<td>{value}</td>\n'
        styled_html += '</tr>\n'

    styled_html += '</tbody>\n</table>\n'
    return styled_html


@app.route('/')
def display_dataframe():
    # Apply row styles to the data
    table_1_html = apply_row_styles(df_merged)
    # table_2_html = apply_row_styles(df_hist)

    # Render the template with styled tables
    return render_template(
        "index.html",
        data_1=table_1_html,
        # data_2=table_2_html,
        date=datetime.now().strftime('%Y-%m-%d')
    )


# New route for About Us page
@app.route('/about')
def about():
    return render_template('about.html')

# New route for Contact page
@app.route('/contact')                                        
def contact():
    return render_template('contact.html')

def open_browser():
    webbrowser.open_new_tab("http://127.0.0.1:5000/")

if __name__ == '__main__':
    # Start the Flask app in a new thread so it doesn’t block the web browser
    Timer(1, open_browser).start()
    app.run(debug=True, use_reloader=False)  # Disables auto-reload
    

