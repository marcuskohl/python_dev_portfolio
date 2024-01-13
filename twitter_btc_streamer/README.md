# Twitter BTC Streamer

## Summary
This is a project created using Python which is designed to stream and analyze tweets containing Bitcoin's symbol "$BTC". It uses X/Twitter's API and Python to provide real-time insights into public sentiment and discussions surrounding Bitcoin on X/Twitter.

## Purpose
The primary goal of this project is to focus on how public perception on social media platforms like X/Twitter correlates with crypto market movements.

## Potential Use Cases
- **Market Sentiment Analysis**: Measuring public sentiment towards Bitcoin.
- **Trend Tracking**: Identifying trends/patterns in discussions about Bitcoin.
- **Data Analysis**: Performing data analysis on the collected data to extract insights.
- **Education**: Could be used as a reference project for those learning about API integration, data streaming, and analysis.

## Getting Started

### Prerequisites
Ensure you have Python installed, as well as X/Twitter Developer account with the necessary access (at least Basic) to use Twitter API v2.

### Installation
1. Clone the repository.
2. Install the required dependencies.

### Usage
1. Set up your Twitter API credentials in `twitter_auth.py`.
2. Run `twitter_stream_listener.py` to start streaming tweets.

## Data Sample
Since I do not have Basic access to the Twitter Developer portal ($100/month), I am unable to stream real tweets using the Twitter API. So for demonstration purposes, a sample CSV file (`data/sample_tweets.csv`) is included in the project. This file shows the structure and format of the data that would be extracted and stored by the script. The CSV contains the following columns:
- `tweet_text`: The content of the tweet.
- `user_id`: The user ID of the tweet author.
- `tweet_time`: The timestamp when the tweet was posted.

## Contributing
Contributions to the project are welcome.

## License
This project is licensed under the MIT License.
