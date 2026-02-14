"""
Domain-specific entity alias dictionary for prediction markets.

Maps canonical names to known aliases.  Used by EntityComparator for
cross-venue entity matching where the same entity has different surface
forms (e.g. "Bitcoin" ↔ "BTC", "Kansas City Chiefs" ↔ "Chiefs").

Maintenance:
    Add new entries to ``_RAW_ALIASES`` below.  The module-level
    ``ALIAS_LOOKUP`` dict is rebuilt on import — no restart needed
    beyond reloading the module.
"""

from typing import Dict, List

# ──────────────────────────────────────────────────────────────────────
# Each key is the canonical form; values are all known aliases (lower-
# case).  The canonical form itself is implicitly included at build time.
# ──────────────────────────────────────────────────────────────────────
_RAW_ALIASES: Dict[str, List[str]] = {
    # ── Crypto ────────────────────────────────────────────────────
    "Bitcoin":    ["btc", "bitcoin", "xbt"],
    "Ethereum":   ["eth", "ethereum", "ether"],
    "Solana":     ["sol", "solana"],
    "Dogecoin":   ["doge", "dogecoin"],
    "Cardano":    ["ada", "cardano"],
    "Ripple":     ["xrp", "ripple"],
    "Litecoin":   ["ltc", "litecoin"],
    "Polkadot":   ["dot", "polkadot"],
    "Avalanche":  ["avax", "avalanche"],
    "Chainlink":  ["link", "chainlink"],
    "Polygon":    ["matic", "polygon"],
    "Tether":     ["usdt", "tether"],
    "USD Coin":   ["usdc", "usd coin"],

    # ── US Politics ───────────────────────────────────────────────
    "Donald Trump":    ["trump", "donald trump", "donald j. trump", "djt"],
    "Joe Biden":       ["biden", "joe biden", "joseph biden"],
    "Kamala Harris":   ["harris", "kamala harris", "kamala", "vp harris"],
    "Ron DeSantis":    ["desantis", "ron desantis"],
    "Vivek Ramaswamy": ["vivek", "ramaswamy", "vivek ramaswamy"],
    "Nikki Haley":     ["haley", "nikki haley"],
    "Robert F. Kennedy Jr.": ["rfk", "rfk jr", "kennedy", "robert kennedy"],
    "Elon Musk":       ["musk", "elon musk", "elon"],
    "JD Vance":        ["vance", "jd vance", "j.d. vance"],

    # ── NFL ───────────────────────────────────────────────────────
    "Kansas City Chiefs":     ["chiefs", "kc chiefs", "kansas city chiefs"],
    "Philadelphia Eagles":    ["eagles", "philly eagles", "philadelphia eagles"],
    "San Francisco 49ers":    ["49ers", "niners", "san francisco 49ers", "sf 49ers"],
    "Dallas Cowboys":         ["cowboys", "dallas cowboys"],
    "Buffalo Bills":          ["bills", "buffalo bills"],
    "Miami Dolphins":         ["dolphins", "miami dolphins"],
    "Baltimore Ravens":       ["ravens", "baltimore ravens"],
    "Detroit Lions":          ["lions", "detroit lions"],
    "Green Bay Packers":      ["packers", "green bay packers"],
    "New York Giants":        ["giants", "ny giants", "new york giants"],
    "New York Jets":          ["jets", "ny jets", "new york jets"],
    "New England Patriots":   ["patriots", "pats", "new england patriots"],
    "Pittsburgh Steelers":    ["steelers", "pittsburgh steelers"],
    "Los Angeles Rams":       ["rams", "la rams", "los angeles rams"],
    "Los Angeles Chargers":   ["chargers", "la chargers", "los angeles chargers"],
    "Tampa Bay Buccaneers":   ["buccaneers", "bucs", "tampa bay buccaneers"],
    "Minnesota Vikings":      ["vikings", "minnesota vikings"],
    "Cincinnati Bengals":     ["bengals", "cincinnati bengals"],
    "Cleveland Browns":       ["browns", "cleveland browns"],
    "Denver Broncos":         ["broncos", "denver broncos"],
    "Houston Texans":         ["texans", "houston texans"],
    "Indianapolis Colts":     ["colts", "indianapolis colts", "indy colts"],
    "Jacksonville Jaguars":   ["jaguars", "jags", "jacksonville jaguars"],
    "Las Vegas Raiders":      ["raiders", "las vegas raiders", "lv raiders"],
    "Seattle Seahawks":       ["seahawks", "seattle seahawks"],
    "Arizona Cardinals":      ["cardinals", "arizona cardinals"],
    "Atlanta Falcons":        ["falcons", "atlanta falcons"],
    "Carolina Panthers":      ["panthers", "carolina panthers"],
    "Chicago Bears":          ["bears", "chicago bears"],
    "New Orleans Saints":     ["saints", "new orleans saints"],
    "Tennessee Titans":       ["titans", "tennessee titans"],
    "Washington Commanders":  ["commanders", "washington commanders"],

    # ── NBA ───────────────────────────────────────────────────────
    "Los Angeles Lakers":       ["lakers", "la lakers", "los angeles lakers"],
    "Golden State Warriors":    ["warriors", "gsw", "golden state warriors"],
    "Boston Celtics":           ["celtics", "boston celtics"],
    "Milwaukee Bucks":          ["bucks", "milwaukee bucks"],
    "Denver Nuggets":           ["nuggets", "denver nuggets"],
    "Phoenix Suns":             ["suns", "phoenix suns"],
    "New York Knicks":          ["knicks", "ny knicks", "new york knicks"],
    "Brooklyn Nets":            ["nets", "brooklyn nets"],
    "Philadelphia 76ers":       ["76ers", "sixers", "philadelphia 76ers"],
    "Miami Heat":               ["heat", "miami heat"],
    "Oklahoma City Thunder":    ["thunder", "okc", "oklahoma city thunder"],
    "Dallas Mavericks":         ["mavericks", "mavs", "dallas mavericks"],
    "Cleveland Cavaliers":      ["cavaliers", "cavs", "cleveland cavaliers"],
    "Sacramento Kings":         ["kings", "sacramento kings"],
    "Minnesota Timberwolves":   ["timberwolves", "wolves", "twolves",
                                 "minnesota timberwolves"],
    "Toronto Raptors":          ["raptors", "toronto raptors"],
    "Chicago Bulls":            ["bulls", "chicago bulls"],
    "Indiana Pacers":           ["pacers", "indiana pacers"],
    "Orlando Magic":            ["magic", "orlando magic"],
    "Houston Rockets":          ["rockets", "houston rockets"],
    "San Antonio Spurs":        ["spurs", "san antonio spurs"],
    "Memphis Grizzlies":        ["grizzlies", "memphis grizzlies"],
    "New Orleans Pelicans":     ["pelicans", "new orleans pelicans"],
    "Portland Trail Blazers":   ["trail blazers", "blazers",
                                 "portland trail blazers"],
    "Utah Jazz":                ["jazz", "utah jazz"],
    "Washington Wizards":       ["wizards", "washington wizards"],
    "Charlotte Hornets":        ["hornets", "charlotte hornets"],
    "Atlanta Hawks":            ["hawks", "atlanta hawks"],
    "Detroit Pistons":          ["pistons", "detroit pistons"],
    "Los Angeles Clippers":     ["clippers", "la clippers",
                                 "los angeles clippers"],

    # ── MLB ───────────────────────────────────────────────────────
    "New York Yankees":     ["yankees", "nyy", "ny yankees", "new york yankees"],
    "New York Mets":        ["mets", "ny mets", "new york mets"],
    "Los Angeles Dodgers":  ["dodgers", "la dodgers", "los angeles dodgers"],
    "Boston Red Sox":       ["red sox", "boston red sox"],
    "Chicago Cubs":         ["cubs", "chicago cubs"],
    "Chicago White Sox":    ["white sox", "chicago white sox"],
    "Houston Astros":       ["astros", "houston astros"],
    "Atlanta Braves":       ["braves", "atlanta braves"],
    "Philadelphia Phillies": ["phillies", "phils", "philadelphia phillies"],
    "San Diego Padres":     ["padres", "san diego padres"],
    "Texas Rangers":        ["rangers", "texas rangers"],
    "San Francisco Giants": ["sf giants", "san francisco giants"],
    "St. Louis Cardinals":  ["cardinals", "stl cardinals",
                             "st. louis cardinals", "st louis cardinals"],
    "Seattle Mariners":     ["mariners", "seattle mariners"],
    "Baltimore Orioles":    ["orioles", "baltimore orioles"],
    "Cleveland Guardians":  ["guardians", "cleveland guardians"],
    "Milwaukee Brewers":    ["brewers", "milwaukee brewers"],
    "Tampa Bay Rays":       ["rays", "tampa bay rays"],
    "Minnesota Twins":      ["twins", "minnesota twins"],
    "Detroit Tigers":       ["tigers", "detroit tigers"],
    "Toronto Blue Jays":    ["blue jays", "jays", "toronto blue jays"],

    # ── NHL ───────────────────────────────────────────────────────
    "Toronto Maple Leafs":   ["maple leafs", "leafs", "toronto maple leafs"],
    "Montreal Canadiens":    ["canadiens", "habs", "montreal canadiens"],
    "Edmonton Oilers":       ["oilers", "edmonton oilers"],
    "New York Rangers":      ["ny rangers", "new york rangers"],
    "Florida Panthers":      ["florida panthers"],
    "Vegas Golden Knights":  ["golden knights", "vgk", "vegas golden knights"],
    "Colorado Avalanche":    ["avalanche", "avs", "colorado avalanche"],
    "Carolina Hurricanes":   ["hurricanes", "canes", "carolina hurricanes"],
    "Dallas Stars":          ["stars", "dallas stars"],
    "New York Islanders":    ["islanders", "ny islanders",
                              "new york islanders"],
    "Tampa Bay Lightning":   ["lightning", "bolts", "tampa bay lightning"],
    "Winnipeg Jets":         ["jets", "winnipeg jets"],
    "Boston Bruins":         ["bruins", "boston bruins"],
    "Pittsburgh Penguins":   ["penguins", "pens", "pittsburgh penguins"],

    # ── Soccer ────────────────────────────────────────────────────
    "Manchester United": ["man utd", "man united", "manchester utd", "mufc"],
    "Manchester City":   ["man city", "mcfc", "manchester city"],
    "Liverpool":         ["liverpool fc", "lfc"],
    "Chelsea":           ["chelsea fc", "cfc"],
    "Arsenal":           ["arsenal fc", "afc", "the gunners"],
    "Tottenham Hotspur": ["tottenham", "spurs", "thfc"],
    "Real Madrid":       ["real madrid cf", "rmcf"],
    "Barcelona":         ["barca", "fc barcelona", "fcb"],
    "Paris Saint-Germain": ["psg", "paris sg", "paris saint germain"],
    "Bayern Munich":     ["bayern", "fc bayern", "bayern munich"],
    "Inter Milan":       ["inter", "internazionale", "inter milan"],
    "Juventus":          ["juve", "juventus fc"],
    "AC Milan":          ["milan", "ac milan", "rossoneri"],
    "Borussia Dortmund": ["dortmund", "bvb", "borussia dortmund"],

    # ── Tennis ────────────────────────────────────────────────────
    "Novak Djokovic":  ["djokovic", "novak", "nole"],
    "Carlos Alcaraz":  ["alcaraz", "carlos alcaraz"],
    "Jannik Sinner":   ["sinner", "jannik sinner"],
    "Coco Gauff":      ["gauff", "coco gauff"],
    "Iga Swiatek":     ["swiatek", "iga swiatek"],
    "Rafael Nadal":    ["nadal", "rafa", "rafael nadal"],
    "Roger Federer":   ["federer", "roger federer"],

    # ── Institutions / Agencies ───────────────────────────────────
    "Federal Reserve":       ["fed", "the fed", "fomc", "federal reserve"],
    "European Central Bank": ["ecb", "european central bank"],
    "Supreme Court":         ["scotus", "supreme court", "the court"],
    "S&P 500":               ["s&p", "s&p 500", "sp500", "spx"],
    "Nasdaq":                ["nasdaq", "qqq", "nasdaq composite"],
    "Dow Jones":             ["dow", "djia", "dow jones"],
    "Bank of England":       ["boe", "bank of england"],
    "Bank of Japan":         ["boj", "bank of japan"],
    "SEC":                   ["sec", "securities and exchange commission"],
}


def build_alias_lookup() -> Dict[str, str]:
    """
    Build a flat lookup: ``lowercase alias → canonical name``.

    Every alias (and the canonical name itself, lowered) maps to
    the same canonical form.  Used for O(1) matching in
    :class:`EntityComparator`.
    """
    lookup: Dict[str, str] = {}
    for canonical, aliases in _RAW_ALIASES.items():
        canonical_lower = canonical.lower()
        # Map canonical form to itself
        lookup[canonical_lower] = canonical
        for alias in aliases:
            lookup[alias.lower()] = canonical
    return lookup


# Module-level singleton — built once on import.
ALIAS_LOOKUP: Dict[str, str] = build_alias_lookup()

