from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document


class SmartWebLoader:
    chunked = True

    def __init__(self, urls: list, **kwargs):
        if isinstance(urls, str):
            urls = [urls]
        self.urls = urls

    def _parse_page(self, url: str) -> Document:
        # Get url parts:
        parsed_url = urlparse(url)
        url_parts = parsed_url.path.rsplit("/", 4)

        # Get html from web url:
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")

        # Get titles:
        titles_span = soup.find_all("span", class_="cmp-accordion__title")
        titles = [title.text for title in titles_span]

        # Get answers:
        answers_div = soup.find_all("div", class_="cmp-text")
        # answers = [answer.encode_contents() for answer in answers_div]
        answers = [answer.get_text() for answer in answers_div]

        # Get hyperlinks to content:
        specific_links_button = soup.find_all("button", class_="cmp-accordion__button")
        specific_links = [
            url + "#" + button.attrs["id"] for button in specific_links_button
        ]

        chunks = []
        for title, answer, specific_link in zip(titles, answers, specific_links):
            content = f"Question: {title}\nAnswer: {answer}\n"
            full_title = f"{url_parts[-4]}/{url_parts[-3]}/{url_parts[-2]}/{title}"
            metadata = {
                "service": url_parts[-4],
                "topic": url_parts[-3],
                "subtopic": url_parts[-2],
                "section": url_parts[-1].removesuffix(".html"),
                "title": full_title,
                "source": specific_link,
            }
            chunks.append(Document(page_content=content, metadata=metadata))

        return chunks

    def load(self):
        docs = []
        for url in self.urls:
            docs.extend(self._parse_page(url))
        return docs
