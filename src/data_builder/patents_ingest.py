import os
import base64
from datetime import date
from typing import List
from pathlib import Path

import requests
import pandas as pd
import xml.etree.ElementTree as ET
from dotenv import load_dotenv

load_dotenv()


class PatentBiblioClient:
    """
    EPO OPS client focused on bibliographic data (title, abstract, etc.)
    for SDA-related patents in a given time window.
    """

    TOKEN_URL = "https://ops.epo.org/3.2/auth/accesstoken"
    SEARCH_URL = "https://ops.epo.org/3.2/rest-services/published-data/search"
    BIBLIO_URL = (
        "https://ops.epo.org/3.2/rest-services/published-data/publication/docdb/"
        "{country}.{doc_number}/biblio"
    )
    EX_NS = "http://www.epo.org/exchange"
    OPS_NS = "http://ops.epo.org"

    def __init__(
        self,
        customer_key: str | None = None,
        customer_secret: str | None = None,
        default_keywords: List[str] | None = None,
        save=False,
    ):
        self.customer_key = customer_key or os.environ.get("PATENT_CUSTOMER_KEY")
        self.customer_secret = customer_secret or os.environ.get("PATENT_CUSTOMER_SECRET_KEY")
        self.save = save

        if not self.customer_key or not self.customer_secret:
            raise ValueError(
                "PATENT_CUSTOMER_KEY and PATENT_CUSTOMER_SECRET_KEY must be set "
                "in environment or passed to PatentBiblioClient."
            )

        self.default_keywords = default_keywords or [
            "air purifier",
            "air fryer",
            "coffee machine",
            "espresso machine",
            "vacuum cleaner",
            "robot vacuum",
        ]

        self.access_token: str | None = None

    def _get_access_token(self) -> str:
        """
        Obtain an OAuth access token from EPO OPS using consumer key + secret.
        """
        if self.access_token:
            return self.access_token

        auth_str = f"{self.customer_key}:{self.customer_secret}"
        b64 = base64.b64encode(auth_str.encode("utf-8")).decode("utf-8")
        headers = {
            "Authorization": f"Basic {b64}",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {"grant_type": "client_credentials"}

        resp = requests.post(self.TOKEN_URL, data=data, headers=headers, timeout=20)
        resp.raise_for_status()

        token_json = resp.json()
        token = token_json.get("access_token")
        if not token:
            raise RuntimeError(f"Could not obtain OPS access token, response: {token_json}")

        self.access_token = token
        return token

    def _auth_headers(self) -> dict:
        """
        Build headers for OPS API calls with Bearer token.
        """
        token = self._get_access_token()
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/xml",
        }

    def _build_query_string(self, keywords: List[str], start_year: int, end_year: int) -> str:
        """
        Build the CQL `q` parameter for OPS search.
        """
        keyword_part = " OR ".join([f"\"{kw}\"" for kw in keywords])
        start_date = f"{start_year}0101"
        end_date = f"{end_year}1231"
        date_part = f"pd={start_date}-{end_date}"

        q = f"({keyword_part}) AND {date_part}"
        return q

    def _request_search(self, q: str, range_start: int = 1, range_end: int = 50) -> str:
        """
        Perform a search request to OPS and return raw XML as text.
        """
        headers = self._auth_headers()
        params = {
            "q": q,
            "Range": f"{range_start}-{range_end}",
        }

        resp = requests.get(self.SEARCH_URL, headers=headers, params=params, timeout=20)
        resp.raise_for_status()

        return resp.text

    def _parse_publication_refs(self, xml_text: str) -> pd.DataFrame:
        """
        Parse OPS biblio-search XML and extract publication references
        (country, doc_number, kind, family_id).
        """
        root = ET.fromstring(xml_text)

        pubs = root.findall(f".//{{{self.OPS_NS}}}publication-reference")

        rows = []
        for pub in pubs:
            family_id = pub.attrib.get("family-id")

            doc_ids = pub.findall(f".//{{{self.EX_NS}}}document-id")
            doc_id = None
            for d in doc_ids:
                if d.attrib.get("document-id-type") == "docdb":
                    doc_id = d
                    break

            if doc_id is None:
                continue

            country = doc_id.findtext(f"{{{self.EX_NS}}}country", default="")
            doc_number = doc_id.findtext(f"{{{self.EX_NS}}}doc-number", default="")
            kind = doc_id.findtext(f"{{{self.EX_NS}}}kind", default="")

            rows.append(
                {
                    "family_id": family_id,
                    "country": country,
                    "doc_number": doc_number,
                    "kind": kind,
                }
            )

        df = pd.DataFrame(rows)
        return df

    def _request_biblio(self, country: str, doc_number: str) -> str:
        """
        Fetch bibliographic data (title, abstract, etc.) for a single patent.
        """
        headers = self._auth_headers()
        url = self.BIBLIO_URL.format(country=country, doc_number=doc_number)

        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()

        return resp.text

    def _parse_biblio_detail_xml(self, xml_text: str) -> dict:
        """
        Parse OPS biblio XML for a single patent and extract
        title, abstract, and publication date.
        """
        root = ET.fromstring(xml_text)

        titles = root.findall(f".//{{{self.EX_NS}}}invention-title")
        title_text = ""
        for t in titles:
            lang = t.attrib.get("lang")
            text = t.text or ""
            if lang == "en":
                title_text = text
                break
            if not title_text:
                title_text = text

        abstracts = root.findall(f".//{{{self.EX_NS}}}abstract")
        abstract_text = ""
        for a in abstracts:
            para = a.find(f".//{{{self.EX_NS}}}p")
            if para is not None and para.text:
                abstract_text = para.text
                break

        pub_date_text = ""
        pub_dates = root.findall(f".//{{{self.EX_NS}}}date")
        for d in pub_dates:
            if d.text:
                pub_date_text = d.text
                break

        return {
            "title": title_text,
            "abstract": abstract_text,
            "publication_date": pub_date_text,
        }

    def fetch_biblio_last_n_years(self, years_back: int, keywords: List[str] | None = None) -> pd.DataFrame:
        """
        Fetch patents from OPS for SDA keywords in the last `years_back` years
        and return a DataFrame with title + abstract ready for NLP.

        Columns:
        - family_id
        - country
        - doc_number
        - kind
        - title
        - abstract
        """
        today = date.today()
        end_year = today.year
        start_year = end_year - years_back

        kw = keywords or self.default_keywords
        q = self._build_query_string(kw, start_year, end_year)

        xml_search = self._request_search(q)
        df_refs = self._parse_publication_refs(xml_search)

        titles = []
        abstracts = []
        pub_dates = []
        for _, row in df_refs.iterrows():
            country = row["country"]
            doc_number = row["doc_number"]
            try:
                xml_biblio = self._request_biblio(country, doc_number)
                detail = self._parse_biblio_detail_xml(xml_biblio)
                titles.append(detail["title"])
                abstracts.append(detail["abstract"])
                pub_dates.append(detail["publication_date"])
            except Exception as e:
                print("Error fetching biblio for", country, doc_number, ":", e)
                titles.append("")
                abstracts.append("")
                pub_dates.append("")

        df_refs["title"] = titles
        df_refs["abstract"] = abstracts
        df_refs["publication_date"] = pub_dates
        df_refs["keywords"] = [self.default_keywords] * len(df_refs)

        if self.save:
            BASE_DIR = Path(__file__).resolve().parent
            output_dir = BASE_DIR / ".." / ".." / "data" / "raw"
            output_dir.mkdir(parents=True, exist_ok=True)

            filename = f"raw-patents-{'-'.join(self.default_keywords)}.csv"
            df_refs.to_csv(output_dir / filename, index=False)

        return df_refs