import praw
import pandas as pd
import time
from datetime import datetime, timedelta

# Initialize Reddit API with your credentials
reddit = praw.Reddit(client_id='UwQGu_BZV3plC9jLCXaRTg',
                     client_secret='pIR4KDYFE0cxjBh73acT7voeSEKm7g',
                     user_agent='LapSent')

# Define the subreddit and keyword you are interested in
subreddit_name = 'Toyota'
keyword = 'cost'

# Create a list to store the filtered headlines
filtered_headlines = []

# Set the time range for submissions (from today to one year ago)
end_time = int(time.time())  # Current epoch time
start_time = end_time - 63072000      # 2 year ago

# Iterate through the submissions in the subreddit within the time range
for submission in reddit.subreddit(subreddit_name).search(f'{keyword}', time_filter='year', limit=None):
    # Check if the keyword is in the title
    if keyword.lower() in submission.title.lower():
        # Add the headline to the list
        filtered_headlines.append(submission.title)

# Convert the list of headlines to a DataFrame
df = pd.DataFrame({'Headline': filtered_headlines})

# Define the CSV file name including the keyword
csv_file_name = f'{subreddit_name}_{keyword}.csv'

# Save the DataFrame to a CSV file with the specified name
df.to_csv(csv_file_name, index=False)

print("Filtered headlines saved")
