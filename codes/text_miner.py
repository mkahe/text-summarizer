from webbrowser import get
from bs4 import BeautifulSoup as bs 
import requests

def get_text(url):
    """Get text from a URL"""
    # Get the HTML
    html = requests.get(url).text
    # Parse the HTML
    soup = bs(html, 'lxml')
    # Get the H1 tag text
    h1 = soup.find('h1').text
    print(h1)
    # Get the all paragraphs text after the h1 tag and before h2 tag
    p = soup.find('h1').find_next_siblings('p')

    # p = soup.find('h1').find_all_next('p')
    print(p)


if __name__ == "__main__":
    # Get the text
    text = get_text("https://en.wikipedia.org/wiki/Finland")
    # text = get_text("https://en.wikipedia.org/wiki/Oulu")
    # Print the first 1000 characters
    # print(text[:1000])