from bs4 import BeautifulSoup as bs 
import requests

def get_text(url):
    """Get text from a URL"""
    # Get the HTML
    html = requests.get(url).text
    # Parse the HTML
    soup = bs(html, 'lxml')
    # Get the text
    text = soup.get_text()
    return text

if __name__ == "__main__":
    # Get the text
    text = get_text("https://en.wikipedia.org/wiki/Finland")
    # Print the first 1000 characters
    print(text[:1000])