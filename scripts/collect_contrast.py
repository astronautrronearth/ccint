"""Contrast-domain collection for the WWW evaluation (PREREG: www2027/results/frozen_eval/PREREG.md).

    python scripts/collect_contrast.py --since 2026-10-21 --until 2026-11-05 [--domain nba|crypto|ai]

Collects Bluesky posts for a non-security domain (NBA basketball) into the same `posts` table under a
separate source_key (`bluesky_nba`) and query_version (`nba_v1`), so the cybersecurity corpus and its
checkpoints are untouched. Day-chunked backfill, a fixed number of pages per term and day (the sampling
frame is "the most recent pages per term-day", recorded in collection_runs.query_spec).
"""
import argparse
import sys
from datetime import datetime, timezone

sys.path[:0] = ["src"]
from ccint.collectors.bluesky import BlueskyCollector  # noqa: E402
from ccint import db  # noqa: E402
from ccint.ingest import run_collection  # noqa: E402

# One term per team (city added where the nickname is an ordinary word), plus the league.
NBA_TERMS = [
    "NBA", "Lakers", "Celtics", "Warriors", "Knicks", "Miami Heat", "Bucks", "Nuggets", "Mavericks",
    "Phoenix Suns", "76ers", "Clippers", "Timberwolves", "OKC Thunder", "Cavaliers", "Raptors", "Pistons",
    "Pacers", "Grizzlies", "Pelicans", "Spurs", "Houston Rockets", "Orlando Magic", "Atlanta Hawks",
    "Charlotte Hornets", "Chicago Bulls", "Brooklyn Nets", "Utah Jazz", "Washington Wizards",
    "Sacramento Kings", "Trail Blazers",
]


CRYPTO_TERMS = ["bitcoin", "ethereum", "crypto", "solana", "dogecoin", "XRP", "stablecoin", "memecoin", "altcoin",
                "DeFi", "NFT", "Binance", "Coinbase"]
AI_TERMS = ["ChatGPT", "OpenAI", "LLM", "generative AI", "Anthropic", "Claude AI", "Gemini AI", "AI model",
            "machine learning", "Copilot", "deepfake", "AI agent", "Hugging Face"]
# PREREG_DOSE2 (2026-11-07): six further domains, same frame
CLIMATE_TERMS = ["climate change", "global warming", "carbon emissions", "renewable energy", "solar power", "wind power",
                 "net zero", "fossil fuels", "COP31", "climate crisis", "electric vehicle", "heat pump", "wildfire"]
FILM_TERMS = ["box office", "Netflix", "HBO", "Disney+", "Marvel", "official trailer", "film review", "Oscars",
              "season finale", "showrunner", "Letterboxd", "movie theater", "Rotten Tomatoes"]
GAMES_TERMS = ["PlayStation", "Xbox", "Nintendo Switch", "Steam Deck", "video game", "indie game", "speedrun",
               "Elden Ring", "GTA 6", "Minecraft", "Fortnite", "esports", "game review"]
SOCCER_TERMS = ["Premier League", "Champions League", "Arsenal", "Liverpool FC", "Manchester United", "Man City",
                "Chelsea FC", "Tottenham", "La Liga", "Bundesliga", "Serie A", "MLS", "Real Madrid"]
SPACE_TERMS = ["NASA", "SpaceX", "Starship", "ISS", "astronomy", "telescope", "James Webb", "exoplanet", "rocket launch",
               "Artemis", "Mars rover", "Blue Origin", "aurora"]
STOCKS_TERMS = ["stock market", "S&P 500", "Nasdaq", "Dow Jones", "earnings call", "stocks", "Federal Reserve",
                "interest rates", "inflation", "Tesla stock", "Nvidia stock", "IPO", "dividend"]
# PREREG_DOSE3 (2026-11-07): six test domains for the robust counts fixed in PREREG_DOSE3
MUSIC_TERMS = ["new album", "Spotify", "concert", "music video", "Grammys", "Taylor Swift", "Billboard", "vinyl",
               "setlist", "tour dates", "Bandcamp", "K-pop", "album review"]
BOOKS_TERMS = ["new book", "novel", "reading list", "Goodreads", "audiobook", "booktok", "bookstore", "publishing",
               "book review", "fantasy book", "Booker Prize", "poetry collection", "library"]
F1_TERMS = ["Formula 1", "F1", "Grand Prix", "Verstappen", "Hamilton", "McLaren", "Ferrari F1", "Red Bull Racing",
            "qualifying", "pit stop", "Norris", "Leclerc", "MotoGP"]
TENNIS_TERMS = ["tennis", "ATP", "WTA", "Djokovic", "Alcaraz", "Sinner", "Sabalenka", "Swiatek", "Coco Gauff",
                "Davis Cup", "Billie Jean King Cup", "ATP Finals", "grand slam"]
HEALTH_TERMS = ["vaccine", "measles", "flu season", "COVID", "public health", "CDC", "World Health Organization", "mental health",
                "long covid", "bird flu", "Medicaid", "outbreak", "RFK Jr"]
FOOD_TERMS = ["recipe", "baking", "sourdough", "restaurant", "cooking", "dinner tonight", "vegan", "coffee",
              "Thanksgiving dinner", "ramen", "pizza", "meal prep", "food review"]
DOMAINS = {"nba": (NBA_TERMS, "nba_v1"), "crypto": (CRYPTO_TERMS, "crypto_v1"), "ai": (AI_TERMS, "ai_v1"),
           "climate": (CLIMATE_TERMS, "climate_v1"), "film": (FILM_TERMS, "film_v1"), "games": (GAMES_TERMS, "games_v1"),
           "soccer": (SOCCER_TERMS, "soccer_v1"), "space": (SPACE_TERMS, "space_v1"), "stocks": (STOCKS_TERMS, "stocks_v1"),
           "music": (MUSIC_TERMS, "music_v1"), "books": (BOOKS_TERMS, "books_v1"), "f1": (F1_TERMS, "f1_v1"),
           "tennis": (TENNIS_TERMS, "tennis_v1"), "health": (HEALTH_TERMS, "health_v1"), "food": (FOOD_TERMS, "food_v1")}


class ContrastCollector(BlueskyCollector):
    source_key = "bluesky_nba"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", required=True)
    ap.add_argument("--until", required=True)
    ap.add_argument("--pages", type=int, default=3)
    ap.add_argument("--domain", default="nba", choices=sorted(DOMAINS))
    a = ap.parse_args()
    terms, qv = DOMAINS[a.domain]
    ContrastCollector.source_key = f"bluesky_{a.domain}"
    since = datetime.fromisoformat(a.since).replace(tzinfo=timezone.utc)
    until = datetime.fromisoformat(a.until).replace(tzinfo=timezone.utc)
    with ContrastCollector(terms, query_version=qv, lang="en") as col:
        res = run_collection(col, mode="backfill", since=since, until=until, max_pages_per_term=a.pages,
                             chunk_days=1)
    # RawPost.source_key is a module constant in the Bluesky collector; restamp this run's new rows
    with db.transaction() as c:
        n = c.execute("UPDATE posts SET source_key = %s WHERE first_seen_run = %s AND source_key = 'bluesky'",
                      (ContrastCollector.source_key, res["run_id"])).rowcount
    print(res, "restamped", n)


if __name__ == "__main__":
    main()
